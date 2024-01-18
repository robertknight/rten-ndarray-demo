use image::imageops::{resize, FilterType};
use image::io::Reader as ImageReader;
use ndarray::{s, Array2, Array4};

use std::error::Error;
use std::fs;

use rten::Model;
use rten_tensor::prelude::*;
use rten_tensor::{NdTensor, NdTensorView};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args();
    args.next(); // Skip binary name

    let model_path = args.next().ok_or("missing `model` arg")?;
    let image_path = args.next().ok_or("missing `image` arg")?;

    let model_bytes = fs::read(model_path)?;
    let model =
        Model::load(&model_bytes).map_err(|err| format!("error loading model: {:?}", err))?;

    // Resize to (224x224), which is a common sized used by many ImageNet models.
    let [input_width, input_height] = [224, 224];
    let img = ImageReader::open(image_path)?.decode()?.into_rgb8();
    let img = resize(&img, input_width, input_height, FilterType::Triangle);

    // Read pixels into NCHW tensor and apply standard ImageNet normalization.
    let imagenet_mean = [0.485, 0.456, 0.406];
    let imagenet_std_dev = [0.229, 0.224, 0.225];
    let nchw_array = Array4::from_shape_fn(
        [1, 3, input_height as usize, input_width as usize],
        |(_n, c, y, x)| {
            let value = img.get_pixel(x as u32, y as u32)[c] as f32 / 255.0;
            (value - imagenet_mean[c]) / imagenet_std_dev[c]
        },
    );
    let nchw_tensor: NdTensorView<f32, 4> = nchw_array
        .as_slice()
        .map(|slice| {
            let shape: [usize; 4] = nchw_array.shape().try_into().unwrap();
            NdTensorView::from_slice(slice, shape, None).expect("incorrect slice length")
        })
        .expect("failed to convert ndarray");

    // Run model inference.
    let output = model
        .run_one(nchw_tensor.as_dyn().into(), None)
        .map_err(|err| format!("model run failed {:?}", err))?;
    let output: NdTensor<f32, 2> = output.try_into()?; // (batch, cls)

    // Convert output back into an ndarray array and find the best score.
    // Note the scores are raw and not directly usable as probabilities. To get
    // probabilities you'd need to apply softmax to the output. See examples
    // in the RTen repo for how to do that.
    let logits_array = Array2::from_shape_vec(output.shape(), output.into_data())?;
    let (top_cls, top_score) = logits_array
        .slice(s![0, ..])
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_cls_a, score_a), (_cls_b, score_b)| score_a.total_cmp(score_b))
        .unwrap();

    let labels_json = fs::read_to_string("imagenet-simple-labels.json")?;
    let labels: Vec<String> =
        serde_json::from_str(&labels_json)?;
    let top_label = labels.get(top_cls).map(|s| s.as_str()).unwrap_or("unknown");

    println!("Top class: {} (score: {})", top_label, top_score);

    Ok(())
}
