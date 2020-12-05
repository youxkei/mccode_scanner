extern crate image;
mod scanner;

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(
    name = "mccode_scanner",
    about = "Scans a MC code image and extracts its data."
)]
struct Opt {
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    #[structopt(long)]
    intermediate: bool,
}

fn main() {
    let opt = Opt::from_args();
    println!(
        "{:?}",
        scanner::scan(image::open(opt.input).unwrap(), opt.intermediate)
    );
}
