use std::time::Instant;

use crseo::{
    calibrations::{Mirror, Segment},
    Atmosphere, Calibration, Diffractive, FromBuilder, Geometric, Gmt, ShackHartmann,
};
use dos_actors::{
    clients::{
        arrow_client::{Arrow, Get},
        ceo::{
            DetectorFrame, M2rxy, OpticalModel, OpticalModelOptions, SegmentPiston, SensorData,
            ShackHartmannOptions, WfeRms,
        },
        Integrator,
    },
    prelude::*,
};
use nalgebra as na;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let atm = OpticalModelOptions::Atmosphere {
        builder: Atmosphere::builder(),
        time_step: 5e-3,
    };

    let n_step = 15;
    let mut timer: Initiator<_> = Timer::new(n_step).into();

    let n_px_frame = 512;

    let imgr = OpticalModelOptions::ShackHartmann {
        options: ShackHartmannOptions::Diffractive(ShackHartmann::<Diffractive>::builder()),
        flux_threshold: 0f64,
    };

    let gmt_builder = Gmt::builder();
    let mut optical_model: Actor<_> = (
        OpticalModel::builder()
            .gmt(gmt_builder.clone())
            .options(vec![imgr, atm.clone()])
            .build()?,
        "On-axis GMT",
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

    println!("Computing AO poke matrix & pseudo-inverse");
    let now = Instant::now();
    let mut calib = Calibration::new(
        &adaptive_optics.gmt,
        &adaptive_optics.src,
        wfs_builder.clone(),
    );
    let poker = vec![Some(vec![(Mirror::M2, vec![Segment::Rxyz(1e-6, Some(0..2))])]); 7];
    calib.calibrate(
        poker,
        crseo::calibrations::ValidLensletCriteria::OtherSensor(
            adaptive_optics.sensor.as_mut().unwrap(),
        ),
    );
    let poke: Vec<f64> = calib.poke.into();
    let n_mode = 14;
    let poke_mat = na::DMatrix::from_column_slice(poke.len() / n_mode, n_mode, &poke);
    let pinv_poke_mat = poke_mat
        .pseudo_inverse(0.)
        .expect("Failed to compute poke matrix pseudo-inverse");
    println!(
        "pseudo-inverse {:?} computed in {}ms",
        pinv_poke_mat.shape(),
        now.elapsed().as_millis()
    );
    adaptive_optics.sensor_matrix_transform(pinv_poke_mat);

    let mut ao_actor: Actor<_> = (adaptive_optics, "Adaptive Optics").into();

    let logging = Arrow::builder(n_step).filename("logs").build().into_arcx();
    let mut logs: Terminator<_> = Actor::new(logging.clone());

    let ao_logging = Arrow::builder(n_step)
        .filename("ao_logs")
        .build()
        .into_arcx();
    let mut ao_logs: Terminator<_> = Actor::new(ao_logging.clone());

    timer
        .add_output()
        .multiplex(2)
        .build::<Tick>()
        .into_input(&mut optical_model)
        .into_input(&mut ao_actor)
        .confirm()?;
    optical_model
        .add_output()
        .build::<WfeRms>()
        .log(&mut logs)
        .await;
    optical_model
        .add_output()
        .build::<SegmentPiston>()
        .log(&mut logs)
        .await;
    optical_model
        .add_output()
        .build::<DetectorFrame>()
        .logn(&mut logs, n_px_frame * n_px_frame)
        .await;

    ao_actor
        .add_output()
        .build::<WfeRms>()
        .log(&mut ao_logs)
        .await;

    let mut integrator: Actor<_> = Integrator::new(n_mode).gain(0.5).into();
    ao_actor
        .add_output()
        .build::<SensorData>()
        .into_input(&mut integrator);
    integrator
        .add_output()
        .bootstrap()
        .build::<M2rxy>()
        .into_input(&mut ao_actor)
        .confirm()?;

    Model::new(vec![
        Box::new(timer),
        Box::new(optical_model),
        Box::new(ao_actor),
        Box::new(integrator),
        Box::new(logs),
        Box::new(ao_logs),
    ])
    .name("dos101")
    .flowchart()
    .check()?
    .run()
    .wait()
    .await?;

    let data: Vec<Vec<f64>> = (*logging.lock().await).get("WfeRms")?;
    println!("{:?}", data);
    let data: Vec<Vec<f64>> = (*ao_logging.lock().await).get("WfeRms")?;
    println!("{:?}", data);
    Ok(())
}
