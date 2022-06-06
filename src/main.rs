use crseo::{Diffractive, FromBuilder, ShackHartmann};
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
    let mut timer: Initiator<_> = Timer::new(5).into();
    let n_px_frame = 64;
    let mut optical_model: Actor<_> = (
        OpticalModel::builder()
            .options(vec![OpticalModelOptions::ShackHartmann {
                options: ShackHartmannOptions::Diffractive(
                    ShackHartmann::<Diffractive>::builder().detector(
                        n_px_frame,
                        None,
                        Some(4),
                        None,
                    ),
                ),
                flux_threshold: 0f64,
            }])
            .build()?,
        "On-axis GMT",
    )
        .into();
    let logging = Arrow::builder(5).build().into_arcx();
    let mut logs: Terminator<_> = Actor::new(logging.clone());

    timer
        .add_output()
        .build::<Tick>()
        .into_input(&mut optical_model);
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

    Model::new(vec![
        Box::new(timer),
        Box::new(optical_model),
        Box::new(logs),
    ])
    .name("dos101")
    .flowchart()
    .check()?
    .run()
    .wait()
    .await?;

    let data: Vec<Vec<f64>> = (*logging.lock().await).get("WfeRms")?;
    println!("{:?}", data);

    Ok(())
}
