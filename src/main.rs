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
use vec_box::vec_box;

pub struct Reconstructor {
    mat: na::DMatrix<f64>,
    u: na::DVector<f64>,
    y: na::DVector<f64>,
}
impl Reconstructor {
    pub fn new(mat: na::DMatrix<f64>) -> Self {
        let (n_y, n_u) = mat.shape();
        Self {
            mat,
            u: na::DVector::zeros(n_u),
            y: na::DVector::zeros(n_y),
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
        let mut data = vec![0f64];
        data.extend_from_slice(self.y.as_mut_slice());
        Some(Arc::new(Data::new(data)))
    }
}

#[derive(UID)]
#[uid(data = "Vec<f32>")]
enum ResidualWavefront {}
impl Write<Vec<f32>, ResidualWavefront> for OpticalModel {
    fn write(&mut self) -> Option<Arc<Data<ResidualWavefront>>> {
        let mut data: Arc<Data<Wavefront>> = self.write()?;
        let inner = Arc::get_mut(&mut data)?;
        Some(Arc::new(inner.into()))
    }
}
impl Size<ResidualWavefront> for OpticalModel {
    fn len(&self) -> usize {
        <Self as Size<Wavefront>>::len(self)
    }
}
#[derive(UID)]
#[uid(data = "Vec<f32>")]
enum ReconWavefront {}
impl Write<Vec<f32>, ReconWavefront> for OpticalModel {
    fn write(&mut self) -> Option<Arc<Data<ReconWavefront>>> {
        let mut data: Arc<Data<Wavefront>> = self.write()?;
        let inner = Arc::get_mut(&mut data)?;
        Some(Arc::new(inner.into()))
    }
}
impl Size<ReconWavefront> for OpticalModel {
    fn len(&self) -> usize {
        <Self as Size<Wavefront>>::len(self)
    }
}

#[derive(UID)]
enum ResidualWfeRms {}
impl Write<Vec<f64>, ResidualWfeRms> for OpticalModel {
    fn write(&mut self) -> Option<Arc<Data<ResidualWfeRms>>> {
        let mut data: Arc<Data<WfeRms>> = self.write()?;
        let inner = Arc::get_mut(&mut data)?;
        Some(Arc::new(inner.into()))
    }
}
impl Size<ResidualWfeRms> for OpticalModel {
    fn len(&self) -> usize {
        <Self as Size<WfeRms>>::len(self)
    }
}

#[derive(UID)]
enum SegmentResidualWfeRms {}
impl Write<Vec<f64>, SegmentResidualWfeRms> for OpticalModel {
    fn write(&mut self) -> Option<Arc<Data<SegmentResidualWfeRms>>> {
        let mut data: Arc<Data<SegmentWfeRms>> = self.write()?;
        let inner = Arc::get_mut(&mut data)?;
        Some(Arc::new(inner.into()))
    }
}
impl Size<SegmentResidualWfeRms> for OpticalModel {
    fn len(&self) -> usize {
        <Self as Size<SegmentWfeRms>>::len(self)
    }
}

#[derive(UID)]
enum SegmentResidualPiston {}
impl Write<Vec<f64>, SegmentResidualPiston> for OpticalModel {
    fn write(&mut self) -> Option<Arc<Data<SegmentResidualPiston>>> {
        let mut data: Arc<Data<SegmentPiston>> = self.write()?;
        let inner = Arc::get_mut(&mut data)?;
        Some(Arc::new(inner.into()))
    }
}
impl Size<SegmentResidualPiston> for OpticalModel {
    fn len(&self) -> usize {
        <Self as Size<SegmentPiston>>::len(self)
    }
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let atm = OpticalModelOptions::Atmosphere {
        builder: Atmosphere::builder(),
        time_step: 1e-3,
    };

    let n_step = 250;
    let mut timer: Initiator<_> = Timer::new(n_step).into();

    let n_px_frame = 512;

    let imgr = OpticalModelOptions::ShackHartmann {
        options: ShackHartmannOptions::Diffractive(ShackHartmann::<Diffractive>::builder()),
        flux_threshold: 0f64,
    };

    let m2_n_mode = 100;
    let gmt_builder = Gmt::builder().m2("Karhunen-Loeve", m2_n_mode);
    let optical_model = OpticalModel::builder()
        .gmt(gmt_builder.clone())
        .options(vec![imgr, atm.clone()])
        .build()?
        .into_arcx();
    let mut on_axis: Actor<_> =
        Actor::new(optical_model.clone()).name("On-axis GMT\nw/ Atmosphere");
    let mut gmt: Actor<_> = (
        OpticalModel::builder().gmt(gmt_builder.clone()).build()?,
        "On-axis GMT\nw/o Atmosphere",
    )
        .into();

    let wfs_builder = ShackHartmann::<Geometric>::builder().lenslet_array(60, 8, 25.5 / 60.);
    let wfs = OpticalModelOptions::ShackHartmann {
        options: ShackHartmannOptions::Geometric(wfs_builder.clone()),
        flux_threshold: 0.8f64,
    };
    let mut adaptive_optics = OpticalModel::builder()
        .gmt(gmt_builder)
        .options(vec![wfs, atm])
        .build()?;

    let calib_path = Path::new(format!("pinv_poke_{}.bin", m2_n_mode));
    let n_mode = (m2_n_mode - 1) * 7;
    let pinv_poke_mat: DMatrix<f64> = if calib_path.is_file() {
        println!("Loading pseudo-inverse from {:?}", calib_path);
        let mat: Vec<f64> = bincode::deserialize_from(File::open(calib_path)?)?;
        DMatrix::from_column_slice(n_mode, mat.len() / n_mode, &mat)
    } else {
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
    }; //   adaptive_optics.sensor_matrix_transform(pinv_poke_mat);

    let mut ao_actor: Actor<_> = (adaptive_optics, "Adaptive Optics").into();

    let logging = Arrow::builder(n_step).build().into_arcx();
    let mut logs: Terminator<_> = Actor::new(logging.clone()).name("Logs");

    /*     let ao_logging = Arrow::builder(n_step)
        .filename("ao_logs")
        .build()
        .into_arcx();
    let mut ao_logs: Terminator<_> = Actor::new(ao_logging.clone()); */

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
    on_axis
        .add_output()
        .build::<Wavefront>()
        .log(&mut logs)
        .await;
    gmt.add_output()
        .build::<ReconWavefront>()
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
    ao_actor
        .add_output()
        .build::<ResidualWavefront>()
        .log(&mut logs)
        .await;

    let mut reconstructor: Actor<_> =
        (Reconstructor::new(pinv_poke_mat), "M2 modes\nreconstructor").into();
    ao_actor
        .add_output()
        .build::<SensorData>()
        .into_input(&mut reconstructor);

    let mut integrator: Actor<_> = Integrator::new(n_mode).gain(0.5).into();
    reconstructor
        .add_output()
        .build::<M2modesRec>()
        .into_input(&mut integrator);
    integrator
        .add_output()
        .bootstrap()
        .multiplex(2)
        .build::<M2modes>()
        .into_input(&mut ao_actor)
        .into_input(&mut gmt)
        .confirm()?;

    Model::new(vec_box![
        timer,
        on_axis,
        ao_actor,
        reconstructor,
        integrator,
        logs,
        gmt
    ])
    .name("dos101")
    .flowchart()
    .check()?
    .run()
    .wait()
    .await?;

    let mut timer: Initiator<_> = Timer::new(0).into();
    let mut on_axis: Actor<_> =
        Actor::new(optical_model.clone()).name("On-axis GMT\nw/ Atmosphere");
    let logging = Arrow::builder(1).filename("frame").build().into_arcx();
    let mut logs: Terminator<_> = Actor::new(logging.clone()).name("Logs");

    timer.add_output().build::<Tick>().into_input(&mut on_axis);
    on_axis
        .add_output()
        .bootstrap()
        .build::<DetectorFrame>()
        .logn(&mut logs, n_px_frame * n_px_frame)
        .await;
    Model::new(vec_box!(timer, on_axis, logs))
        .check()?
        .run()
        .wait()
        .await?;

    Ok(())
}
