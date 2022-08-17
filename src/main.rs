use std::{fs::File, path::Path, sync::Arc, time::Instant};

use crseo::{
    calibrations::{Mirror, Segment},
    Atmosphere, Calibration, Diffractive, FromBuilder, Geometric, Gmt, ShackHartmann,
};
use dos_actors::{
    clients::{
        arrow_client::Arrow,
        ceo::{
            DetectorFrame, M2modes, OpticalModel, OpticalModelOptions, SegmentPiston,
            SegmentWfeRms, SensorData, ShackHartmannOptions, Wavefront, WfeRms,
        },
        Integrator,
    },
    io::{Data, Read, Write},
    prelude::*,
    Size, Update,
};
use na::DMatrix;
use nalgebra as na;
use skyangle::SkyAngle;
use uid_derive::UID;
use vec_box::vec_box;

pub struct Reconstructor {
    mat: na::DMatrix<f64>,
    u: na::DVector<f64>,
    y: na::DVector<f64>,
    n_y: usize,
}
impl Reconstructor {
    pub fn new(mat: na::DMatrix<f64>) -> Self {
        let (n_y, n_u) = mat.shape();
        println!("Reconstructor {:?}", (n_y, n_u));
        Self {
            mat,
            u: na::DVector::zeros(n_u),
            y: na::DVector::zeros(n_y),
            n_y,
        }
    }
}
impl Update for Reconstructor {
    fn update(&mut self) {
        self.y = &self.mat * &self.u;
    }
}
impl Read<Vec<f64>, SensorData> for Reconstructor {
    fn read(&mut self, data: Arc<Data<SensorData>>) {
        self.u = na::DVector::from_column_slice(&data);
    }
}

#[derive(UID)]
enum M2modesRec {}
impl Write<Vec<f64>, M2modesRec> for Reconstructor {
    fn write(&mut self) -> Option<Arc<Data<M2modesRec>>> {
        Some(Arc::new(Data::new(
            self.y
                .as_slice()
                .chunks(self.n_y / 7)
                .flat_map(|y| {
                    let mut a = vec![0f64];
                    a.extend_from_slice(y);
                    a
                })
                .collect(),
        )))
    }
}

#[derive(UID)]
#[alias(name = "Wavefront", client = "OpticalModel", traits = "Write,Size")]
enum ResidualWavefront {}
#[derive(UID)]
#[alias(name = "WfeRms", client = "OpticalModel", traits = "Write,Size")]
enum ResidualWfeRms {}
#[derive(UID)]
#[alias(name = "SegmentWfeRms", client = "OpticalModel", traits = "Write,Size")]
enum SegmentResidualWfeRms {}
#[derive(UID)]
#[alias(name = "SegmentPiston", client = "OpticalModel", traits = "Write,Size")]
enum SegmentResidualPiston {}
#[derive(UID)]
#[alias(name = "Wavefront", client = "OpticalModel", traits = "Write,Size")]
enum ReconWavefront {}
#[derive(UID)]
#[alias(name = "DetectorFrame", client = "OpticalModel", traits = "Write")]
enum NaturalSeeingImage {}
#[derive(UID)]
#[alias(name = "DetectorFrame", client = "OpticalModel", traits = "Write")]
enum DiffractionLimitedImage {}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Atmosphere model
    let atm = OpticalModelOptions::Atmosphere {
        builder: Atmosphere::builder().ray_tracing(
            25.5,
            1020,
            SkyAngle::Arcminute(1f32).to_radians(),
            3f32,
            Some("ngao_atmosphere.bin".to_string()),
            Some(1),
        ),
        time_step: 1e-3,
    };

    // Simulation timer
    let n_step = 2000;
    let mut timer: Initiator<_> = Timer::new(n_step).into();

    let n_px_frame = 512;

    // Imaging camera
    let imgr = OpticalModelOptions::ShackHartmann {
        options: ShackHartmannOptions::Diffractive(ShackHartmann::<Diffractive>::builder()),
        flux_threshold: 0f64,
    };

    // GMT model
    let m2_n_mode = 200;
    let n_mode = m2_n_mode * 7;
    let gmt_builder = Gmt::builder().m2("Karhunen-Loeve", m2_n_mode);

    // [GMT + Atmosphere] optical model
    let optical_model = OpticalModel::builder()
        .gmt(gmt_builder.clone())
        .options(vec![imgr.clone(), atm.clone()])
        .build()?
        .into_arcx();
    let mut on_axis: Actor<_> =
        Actor::new(optical_model.clone()).name("On-axis GMT\nw/ Atmosphere");
    // [GMT + Atmosphere + AO] optical model
    let science_path = OpticalModel::builder()
        .gmt(gmt_builder.clone())
        .options(vec![imgr, atm.clone()])
        .build()?
        .into_arcx();
    let mut science: Terminator<_> = Actor::new(science_path.clone()).name("Science Path");
    // GMT optical model
    let gmt_model = OpticalModel::builder()
        .gmt(gmt_builder.clone())
        .build()?
        .into_arcx();
    let mut gmt: Actor<_> = Actor::new(gmt_model.clone()).name("On-axis GMT\nw/o Atmosphere");

    // Shack-Hartmann WFS
    let wfs_builder = ShackHartmann::<Geometric>::builder().lenslet_array(60, 8, 25.5 / 60.);
    let wfs = OpticalModelOptions::ShackHartmann {
        options: ShackHartmannOptions::Geometric(wfs_builder.clone()),
        flux_threshold: 0.8f64,
    };
    // Adaptive Optics model
    let mut adaptive_optics = OpticalModel::builder()
        .gmt(gmt_builder)
        .options(vec![wfs, atm])
        .build()?;

    // Poke matrix pseudo-inverse
    let path = format!("pinv_poke_{}.bin", m2_n_mode);
    let calib_path = Path::new(&path);
    let pinv_poke_mat: DMatrix<f64> = if calib_path.is_file() {
        println!("Loading pseudo-inverse from {:?}", calib_path);
        let mat: Vec<f64> = bincode::deserialize_from(File::open(calib_path)?)?;
        let n_mode = (m2_n_mode - 1) * 7;
        DMatrix::from_column_slice(n_mode, mat.len() / n_mode, &mat)
    } else {
        // Computing & saving
        println!("Computing AO poke matrix & pseudo-inverse");
        let now = Instant::now();
        let mut calib = Calibration::new(
            &adaptive_optics.gmt,
            &adaptive_optics.src,
            wfs_builder.clone(),
        );
        let poker = vec![
            Some(vec![(
                Mirror::M2MODES,
                vec![Segment::Modes(1e-6, 1..m2_n_mode)]
            )]);
            7
        ];
        calib.calibrate(
            poker,
            crseo::calibrations::ValidLensletCriteria::OtherSensor(
                adaptive_optics.sensor.as_mut().unwrap(),
            ),
        );
        let poke: Vec<f64> = calib.poke.into();

        let n_mode = (m2_n_mode - 1) * 7;
        let poke_mat = na::DMatrix::from_column_slice(poke.len() / n_mode, n_mode, &poke);
        let svd = poke_mat.svd(false, false);
        let svals = svd.singular_values.as_slice();
        let condn = svals[0] / svals.last().unwrap();
        println!("Condition #: {}", condn);

        let poke_mat = na::DMatrix::from_column_slice(poke.len() / n_mode, n_mode, &poke);
        let pinv_poke_mat = poke_mat
            .pseudo_inverse(0.)
            .expect("Failed to compute poke matrix pseudo-inverse");
        println!(
            "pseudo-inverse {:?} computed in {}ms",
            pinv_poke_mat.shape(),
            now.elapsed().as_millis()
        );
        println!("Saving pseudo-inverse to {:?}", calib_path);
        bincode::serialize_into(File::create(calib_path)?, pinv_poke_mat.as_slice())?;
        pinv_poke_mat
    };

    let adaptive_optics = adaptive_optics.into_arcx();
    let mut ao_actor: Actor<_> = Actor::new(adaptive_optics.clone()).name("Adaptive Optics");

    // Telemetry logs
    //  . WFE terms
    let logging = Arrow::builder(n_step)
        .filename("ngao-logs")
        .build()
        .into_arcx();
    let mut logs: Terminator<_> = Actor::new(logging.clone()).name("Logs");
    //  . Last wavefronts
    let wavefront_logging = Arrow::builder(1)
        .decimation(n_step)
        .filename("ngao-wavefront")
        .build()
        .into_arcx();
    let mut wavefront_logs: Terminator<_> =
        Actor::new(wavefront_logging.clone()).name("Wavefront Logs");
    timer
        .add_output()
        .multiplex(2)
        .build::<Tick>()
        .into_input(&mut on_axis)
        .into_input(&mut ao_actor)
        .confirm()?;
    on_axis.add_output().build::<WfeRms>().log(&mut logs).await;
    on_axis
        .add_output()
        .build::<SegmentWfeRms>()
        .log(&mut logs)
        .await;
    on_axis
        .add_output()
        .build::<SegmentPiston>()
        .log(&mut logs)
        .await;

    ao_actor
        .add_output()
        .build::<ResidualWfeRms>()
        .log(&mut logs)
        .await;
    ao_actor
        .add_output()
        .build::<SegmentResidualWfeRms>()
        .log(&mut logs)
        .await;
    ao_actor
        .add_output()
        .build::<SegmentResidualPiston>()
        .log(&mut logs)
        .await;

    on_axis
        .add_output()
        .build::<Wavefront>()
        .log(&mut wavefront_logs)
        .await;
    gmt.add_output()
        .build::<ReconWavefront>()
        .log(&mut wavefront_logs)
        .await;
    ao_actor
        .add_output()
        .build::<ResidualWavefront>()
        .log(&mut wavefront_logs)
        .await;

    // WFS 2 M2 modes reconstructor
    let mut reconstructor: Actor<_> =
        (Reconstructor::new(pinv_poke_mat), "M2 modes\nreconstructor").into();
    ao_actor
        .add_output()
        .build::<SensorData>()
        .into_input(&mut reconstructor);

    // Control system
    let mut integrator: Actor<_> = Integrator::new(n_mode).gain(0.5).into();
    reconstructor
        .add_output()
        .build::<M2modesRec>()
        .into_input(&mut integrator);
    integrator
        .add_output()
        .bootstrap()
        .multiplex(3)
        .build::<M2modes>()
        .into_input(&mut ao_actor)
        .into_input(&mut gmt)
        .into_input(&mut science)
        .confirm()?;

    let adaptive_optics_system = Model::new(vec_box![
        timer,
        on_axis,
        ao_actor,
        reconstructor,
        integrator,
        logs,
        wavefront_logs,
        gmt,
        science
    ])
    .name("ngao")
    .flowchart()
    .check()?
    .run();

    // Reading out imaging cameras
    let mut timer: Initiator<_> = Timer::new(0).progress().into();
    let mut on_axis: Actor<_> =
        Actor::new(optical_model.clone()).name("On-axis GMT\nw/ Atmosphere");
    let mut science: Actor<_> = Actor::new(science_path.clone()).name("Science Path");
    let logging = Arrow::builder(1).filename("ngao-frame").build().into_arcx();
    let mut logs: Terminator<_> = Actor::new(logging.clone()).name("Logs");

    timer
        .add_output()
        .multiplex(2)
        .build::<Tick>()
        .into_input(&mut on_axis)
        .into_input(&mut science)
        .confirm()?;
    on_axis
        .add_output()
        .bootstrap()
        .build::<NaturalSeeingImage>()
        .logn(&mut logs, n_px_frame * n_px_frame)
        .await;

    science
        .add_output()
        .bootstrap()
        .build::<DiffractionLimitedImage>()
        .logn(&mut logs, n_px_frame * n_px_frame)
        .await;
    let detector_readouts = Model::new(vec_box!(timer, on_axis, science, logs))
        .name("ngao-images")
        .check()?
        .flowchart();

    adaptive_optics_system.wait().await?;
    detector_readouts.run().wait().await?;

    Ok(())
}
