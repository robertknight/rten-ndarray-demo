# rten-ndarray-demo

This is an example showing how to integrate a few popular packages in order to
perform simple ML inference using [RTen](https://github.com/robertknight/rten).

- [image-rs](https://github.com/image-rs) is used to read images from JPEG or
  PNG sources
- [ndarray](https://docs.rs/ndarray/latest/ndarray/) is used to prepare the
  input and post-process the output. You can use RTen's own tensor types for
  this purpose, but many users may be more familiar with ndarray.
- serde_json is used to read label data

## Usage

Clone this repository, then run:

```
$ cargo run -r -- mobilenet.rten cat.jpg
    Finished release [optimized] target(s) in 0.30s
     Running `target/release/rten-mobilenet mobilenet.rten cat.jpg`
Top class: tabby cat (score: 16.08712)
```

## Credits

- MobileNet model retrieved from https://huggingface.co/timm/mobilenetv2_100.ra_in1k
  using the `export-timm-model.py` script in the RTen repo, then converted using
  RTen's ONNX => .rten conversion scripts.

- Cat image from https://en.wikipedia.org/wiki/Tabby_cat

- ImageNet labels from https://github.com/anishathalye/imagenet-simple-labels
