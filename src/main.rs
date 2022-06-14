use crseo::{Atmosphere, Diffractive, FromBuilder, ShackHartmann};
use dos_actors::{
    clients::{
        arrow_client::{Arrow, Get},
        ceo::{
            DetectorFrame, OpticalModel, OpticalModelOptions, SegmentPiston, ShackHartmannOptions,
            WfeRms,
        },
    },
    prelude::*,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut timer: Initiator<_> = Timer::new(3).into();
    let n_px_frame = 512;
    let mut optical_model: Actor<_> = (
        OpticalModel::builder()
            .options(vec![
                OpticalModelOptions::ShackHartmann {
                    options: ShackHartmannOptions::Diffractive(
                        ShackHartmann::<Diffractive>::builder(),
                    ),
                    flux_threshold: 0f64,
                },
                OpticalModelOptions::Atmosphere {
                    builder: Atmosphere::builder(),
                    time_step: 5e-3,
                },
            ])
            .build()?,
        "On-axis GMT",
    )
        .into();
    let mut adaptive_optics: Actor<_> = (
        OpticalModel::builder()
            .options(vec![
                OpticalModelOptions::ShackHartmann {
                    options: ShackHartmannOptions::Diffractive(
                        ShackHartmann::<Diffractive>::builder(),
                    ),
                    flux_threshold: 0f64,
                },
                OpticalModelOptions::Atmosphere {
                    builder: Atmosphere::builder(),
                    time_step: 5e-3,
                },
            ])
            .build()?,
        "Adaptive Optics",
    )
        .into();
    let logging = Arrow::builder(5).build().into_arcx();
    let mut logs: Terminator<_> = Actor::new(logging.clone());

    let ao_logging = Arrow::builder(5).build().into_arcx();
    let mut ao_logs: Terminator<_> = Actor::new(ao_logging.clone());

    timer
        .add_output()
        .multiplex(2)
        .build::<Tick>()
        .into_input(&mut optical_model)
        .into_input(&mut adaptive_optics)
        .confirm()?;
    optical_model
        .add_output()
        .build::<WfeRms>()
        .log(&mut logs)
        .await;
    adaptive_optics
        .add_output()
        .build::<WfeRms>()
        .log(&mut ao_logs)
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

    Model::new(vec![
        Box::new(timer),
        Box::new(optical_model),
        Box::new(adaptive_optics),
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
