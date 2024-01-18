use image::imageops::{resize, FilterType};
use image::io::Reader as ImageReader;

use std::error::Error;
use std::fs;

use rten::{FloatOperators, Model, Operators};
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;

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
    let mut nchw_tensor = NdTensor::zeros([1, 3, input_height as usize, input_width as usize]);
    for y in 0..nchw_tensor.size(2) {
        for x in 0..nchw_tensor.size(3) {
            for c in 0..nchw_tensor.size(1) {
                let value = img.get_pixel(x as u32, y as u32)[c] as f32 / 255.0;
                let value = (value - imagenet_mean[c]) / imagenet_std_dev[c];
                nchw_tensor[[0, c, y, x]] = value;
            }
        }
    }

    // Run model inference.
    let output = model
        .run_one(nchw_tensor.as_dyn().into(), None)
        .map_err(|err| format!("model run failed {:?}", err))?;
    let output: NdTensor<f32, 2> = output.try_into()?; // (batch, cls)
    let (topk_scores, topk_cls) =
        output
            .softmax(1)?
            .topk(5, Some(1), true /* largest */, true /* sorted */)?;

    // Convert output back into an ndarray array and find the best score.
    // Note the scores are raw and not directly usable as probabilities. To get
    // probabilities you'd need to apply softmax to the output. See examples
    // in the RTen repo for how to do that.
    let labels_json = fs::read_to_string("imagenet-simple-labels.json")?;
    let labels: Vec<String> = serde_json::from_str(&labels_json)?;
    for (score, &cls) in topk_scores.iter().zip(topk_cls.iter()) {
        let label = labels
            .get(cls as usize)
            .map(|s| s.as_str())
            .unwrap_or("unknown");
        println!("Class {} score {}", label, score);
    }

    Ok(())
}
