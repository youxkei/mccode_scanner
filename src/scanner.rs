use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use image::imageops::colorops::{grayscale, invert};
use image::{DynamicImage, GrayImage, Luma};

use imageproc::contrast::{otsu_level, threshold_mut};
use imageproc::distance_transform::{distance_transform, Norm};
use imageproc::morphology::{dilate_mut, erode_mut};
use imageproc::stats::histogram;

use disjoint_sets::UnionFindNode;

use maplit::hashset;

#[cfg(test)]
use imageproc::assert_pixels_eq;
#[cfg(test)]
use imageproc::gray_image;
#[cfg(test)]
use maplit::hashmap;

type Pixel = (u32, u32);
type VertexId = u32;
type Weight = u32;

type FourNeighbors = [(Pixel, u8); 4];
type PortPixels = HashMap<Pixel, RefCell<UnionFindNode<u32>>>;
type AdjacencyMap = HashMap<VertexId, RefCell<HashMap<VertexId, Weight>>>;

const BACKGROUND_PIXEL: u8 = 0;
const REMOVING_PIXEL: u8 = 127;
const FOREGROUND_PIXEL: u8 = 255;

const PORT_PIXEL: u8 = 200;
const CROSSING_PIXEL: u8 = 160;
const EDGE_PIXEL: u8 = 127;

const MAX_FOREGROUND_PERCENTAGE: u32 = 35;

pub fn scan(image: DynamicImage, intermediate: bool) -> Result<String, String> {
    let image = grayscale(&image);

    if intermediate {
        image.save("grayscale.png").unwrap();
    }

    let mut binary_image = thresholding(image);
    let opening_number = calculate_opening_number(&binary_image);

    invert(&mut binary_image);
    let binary_image = framing(binary_image);

    if intermediate {
        binary_image.save("binary.png").unwrap();
    }

    check_foreground_percentage(&binary_image)?;

    let vertices_segment_image = open(binary_image.clone(), opening_number);

    if intermediate {
        vertices_segment_image.save("vertices.png").unwrap();
    }

    let skeleton_image = hilditch_thinning(binary_image);

    if intermediate {
        skeleton_image.save("skeleton.png").unwrap();
    }

    let (classified_skeleton_image, port_pixels) =
        classify_edge_pixels(skeleton_image, &vertices_segment_image)?;

    if intermediate {
        classified_skeleton_image
            .save("classified_skeleton.png")
            .unwrap();
    }

    let (classified_skeleton_image, port_pixels) =
        unify_port_pixels(classified_skeleton_image, port_pixels, opening_number)?;

    if intermediate {
        classified_skeleton_image
            .save("modified_classified_skeleton.png")
            .unwrap();
    }

    let adjacency_map = calculate_adjacency_map(&classified_skeleton_image, &port_pixels)?;

    println!("{:?}", adjacency_map);

    let bits = extract_bits(&adjacency_map)?;
    let data = decode_bits(bits);

    Ok(data)
}

fn calculate_opening_number(image: &GrayImage) -> u8 {
    let (width, height) = image.dimensions();
    let distance_image = distance_transform(image, Norm::L1);

    let mut max_distance = 0;

    for y in 0..height {
        for x in 0..width {
            let p0 = distance_image.get_pixel(x, y)[0];

            if p0 > max_distance {
                max_distance = p0;
            }
        }
    }

    max_distance / 2
}

fn thresholding(mut image: GrayImage) -> GrayImage {
    let threshold_level = otsu_level(&image);
    threshold_mut(&mut image, threshold_level);

    image
}

fn framing(mut image: GrayImage) -> GrayImage {
    let (width, height) = image.dimensions();

    for y in 0..height {
        let (first, last) = (0, width - 1);

        image.put_pixel(first, y, Luma([0]));
        image.put_pixel(last, y, Luma([0]))
    }

    for x in 0..width {
        let (first, last) = (0, height - 1);

        image.put_pixel(x, first, Luma([0]));
        image.put_pixel(x, last, Luma([0]))
    }

    image
}

fn check_foreground_percentage(image: &GrayImage) -> Result<(), String> {
    let hist = histogram(image).channels[0];
    let num_foreground_pixels = hist[FOREGROUND_PIXEL as usize];
    let num_pixels = num_foreground_pixels + hist[BACKGROUND_PIXEL as usize];

    // num_foreground_pixels / num_pixels <= MAX_FOREGROUND_PERCENTAGE / 100
    if num_foreground_pixels * 100 <= MAX_FOREGROUND_PERCENTAGE * num_pixels {
        Ok(())
    } else {
        Err(format!(
            "foregroud percentage is too high. percentage: {:.2}, required: percentage <= {}",
            num_foreground_pixels as f64 / num_pixels as f64 * 100.0,
            MAX_FOREGROUND_PERCENTAGE
        ))
    }
}

fn open(mut image: GrayImage, k: u8) -> GrayImage {
    erode_mut(&mut image, Norm::L1, k);

    dilate_mut(&mut image, Norm::L1, k + 2);

    image
}

#[allow(dead_code)]
fn close(image: GrayImage, k: u8) -> GrayImage {
    let mut image = image;

    dilate_mut(&mut image, Norm::L1, k);

    erode_mut(&mut image, Norm::L1, k);

    image
}

fn hilditch_thinning(image: GrayImage) -> GrayImage {
    const OFFSETS: [(i32, i32); 9] = [
        (0, 0),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    const ODD: [usize; 4] = [1, 3, 5, 7];

    fn eight_connectivity(buffer: &[i8; 9]) -> i8 {
        let mut d: [i8; 10] = [0; 10];

        for i in 0..10 {
            let j = if i == 9 { 1 } else { i };

            if buffer[j].abs() == 1 {
                d[i] = 1
            } else {
                d[i] = 0
            }
        }

        let mut sum = 0;
        for i in 0..4 {
            let j = ODD[i];
            sum += d[j] - d[j] * d[j + 1] * d[j + 2]
        }

        sum
    }

    let mut image = image;
    let (width, height) = image.dimensions();
    let mut changed = true;

    while changed {
        changed = false;

        let mut removing_pixels = Vec::new();

        for y in 1..height - 1 {
            'outer: for x in 1..width - 1 {
                let mut buffer: [i8; 9] = [0; 9];

                for i in 0..9 {
                    let (dx, dy) = OFFSETS[i];
                    buffer[i] =
                        match image.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0] {
                            BACKGROUND_PIXEL => 0,
                            FOREGROUND_PIXEL => 1,
                            REMOVING_PIXEL => -1,
                            _ => panic!(),
                        }
                }

                let buffer = buffer;

                // Condition 1
                if buffer[0] != 1 {
                    continue;
                }

                // Condition 2
                {
                    let mut sum = 0;

                    for i in 0..4 {
                        sum += 1 - buffer[ODD[i]].abs()
                    }

                    if sum == 0 {
                        continue;
                    }
                }

                // Condition 3
                {
                    let mut sum = 0;

                    for i in 1..9 {
                        sum += buffer[i].abs()
                    }

                    if sum < 2 {
                        continue;
                    }
                }

                // Condition 4
                {
                    let mut sum = 0;

                    for i in 1..9 {
                        if buffer[i] == 1 {
                            sum += 1
                        }
                    }

                    if sum == 0 {
                        continue;
                    }
                }

                // Condition 5
                if eight_connectivity(&buffer) != 1 {
                    continue;
                }

                // Condition 6
                {
                    for i in 1..9 {
                        if buffer[i] == -1 {
                            let mut copy = buffer;
                            copy[i] = 0;
                            if eight_connectivity(&copy) != 1 {
                                continue 'outer;
                            }
                        }
                    }
                }

                changed = true;
                removing_pixels.push((x, y));
                image.put_pixel(x, y, Luma([REMOVING_PIXEL]))
            }
        }

        for removed_pixel in removing_pixels {
            let (x, y) = removed_pixel;

            image.put_pixel(x, y, Luma([BACKGROUND_PIXEL]))
        }
    }

    image
}

fn classify_edge_pixels(
    skeleton_image: GrayImage,
    vertices_segment_image: &GrayImage,
) -> Result<(GrayImage, PortPixels), String> {
    let mut port_pixel_id = 1;
    let (width, height) = skeleton_image.dimensions();
    let mut skeleton_image = skeleton_image;
    let mut port_pixels = HashMap::new();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            if skeleton_image.get_pixel(x, y)[0] == 0
                || vertices_segment_image.get_pixel(x, y)[0] > 0
            {
                continue;
            }

            let skeleton_neighbors = four_neighbors(&skeleton_image, x, y);
            let num_foreground_skeleton_neighbors =
                num_foreground_four_neighbors(skeleton_neighbors);

            let vertices_segment_neighbors = four_neighbors(vertices_segment_image, x, y);
            let skeleton_neighbors_inside_vertices =
                bit_and_four_neighbors(skeleton_neighbors, vertices_segment_neighbors);
            let num_foreground_skeleton_neighbors_inside_vertices =
                num_foreground_four_neighbors(skeleton_neighbors_inside_vertices);

            if num_foreground_skeleton_neighbors_inside_vertices > 0 {
                skeleton_image.put_pixel(x, y, Luma([PORT_PIXEL]));
                port_pixels.insert((x, y), RefCell::new(UnionFindNode::new(port_pixel_id)));
                port_pixel_id += 1;
            } else if num_foreground_skeleton_neighbors < 2 {
                skeleton_image.put_pixel(x, y, Luma([0]))
            } else if num_foreground_skeleton_neighbors == 2 {
                skeleton_image.put_pixel(x, y, Luma([EDGE_PIXEL]))
            } else if num_foreground_skeleton_neighbors > 2 {
                skeleton_image.put_pixel(x, y, Luma([CROSSING_PIXEL]))
            }
        }
    }

    Ok((skeleton_image, port_pixels))
}

fn bit_and_four_neighbors(lhs: FourNeighbors, rhs: FourNeighbors) -> FourNeighbors {
    [
        (lhs[0].0, lhs[0].1 & rhs[0].1),
        (lhs[1].0, lhs[1].1 & rhs[1].1),
        (lhs[2].0, lhs[2].1 & rhs[2].1),
        (lhs[3].0, lhs[3].1 & rhs[3].1),
    ]
}

fn num_foreground_four_neighbors(neighbors: FourNeighbors) -> u32 {
    let [p2, p4, p6, p8] = neighbors;

    let mut count = 0;
    if p2.1 > 0 {
        count += 1
    }
    if p4.1 > 0 {
        count += 1
    }
    if p6.1 > 0 {
        count += 1
    }
    if p8.1 > 0 {
        count += 1
    }

    return count;
}

fn unify_port_pixels(
    mut skeleton_image: GrayImage,
    mut port_pixels: PortPixels,
    opening_number: u8,
) -> Result<(GrayImage, PortPixels), String> {
    let mut visited = HashSet::new();

    // unifying port pixels belonging to the same vertex
    for (port_pixel, port_pixel_node) in &port_pixels {
        if visited.contains(&*port_pixel) {
            continue;
        }

        let mut next_pixels = Vec::new();
        next_pixels.push(*port_pixel);

        while let Some(pixel) = next_pixels.pop() {
            visited.insert(pixel);

            match port_pixels.get(&pixel) {
                Some(pixel_node) if pixel != *port_pixel => {
                    let mut port_pixel_node = port_pixel_node.borrow_mut();
                    let mut pixel_node = pixel_node.borrow_mut();

                    port_pixel_node.union(&mut pixel_node);
                }

                _ => {
                    let (x, y) = pixel;

                    for neighbor in &four_neighbors(&skeleton_image, x, y) {
                        if neighbor.1 >= PORT_PIXEL && !visited.contains(&neighbor.0) {
                            next_pixels.push(neighbor.0)
                        }
                    }
                }
            }
        }
    }

    // collecting crossing pixel close to some port pixel along edges
    // the distance should be equal or less than opening_number * 2
    let mut crossing_pixels_close_to_port_pixel = vec![];
    let max_distance = opening_number * 2;

    for (port_pixel, port_pixel_node) in &port_pixels {
        for port_pixel_neighbor in &four_neighbors(&skeleton_image, port_pixel.0, port_pixel.1) {
            if !(port_pixel_neighbor.1 == EDGE_PIXEL
                || port_pixel_neighbor.1 == PORT_PIXEL
                || port_pixel_neighbor.1 == CROSSING_PIXEL)
            {
                continue;
            }

            let mut current_pixel = port_pixel_neighbor.0;
            let mut prev_pixel = *port_pixel;

            for _ in 0..max_distance {
                match skeleton_image.get_pixel(current_pixel.0, current_pixel.1)[0] {
                    CROSSING_PIXEL => {
                        crossing_pixels_close_to_port_pixel
                            .push((current_pixel, port_pixel_node.clone()));

                        break;
                    }

                    PORT_PIXEL => break,

                    _ => {}
                }

                let mut moved = false;

                for neighbor in &four_neighbors(&skeleton_image, current_pixel.0, current_pixel.1) {
                    if neighbor.0 != prev_pixel
                        && (neighbor.1 == EDGE_PIXEL
                            || neighbor.1 == PORT_PIXEL
                            || neighbor.1 == CROSSING_PIXEL)
                    {
                        prev_pixel = current_pixel;
                        current_pixel = neighbor.0;

                        moved = true;
                        break;
                    }
                }

                if !moved {
                    return Err(format!(
                        "unify_port_pixels: dead end edge on pixel: {:?}",
                        current_pixel
                    ));
                }
            }
        }
    }

    for (crossing_pixel, port_pixel_node) in crossing_pixels_close_to_port_pixel {
        port_pixels.insert(crossing_pixel, port_pixel_node);
        skeleton_image.put_pixel(crossing_pixel.0, crossing_pixel.1, Luma([PORT_PIXEL]));
    }

    Ok((skeleton_image, port_pixels))
}

#[test]
fn unify_port_pixels_test() {
    const F: u8 = FOREGROUND_PIXEL;
    const P: u8 = PORT_PIXEL;
    const C: u8 = CROSSING_PIXEL;
    const E: u8 = EDGE_PIXEL;

    {
        let skeleton_image = gray_image!(
            0, 0, 0, 0, 0, 0, 0;
            0, F, P, E, P, F, 0;
            0, P, 0, 0, 0, P, 0;
            0, E, 0, 0, 0, E, 0;
            0, P, 0, 0, 0, P, 0;
            0, F, P, E, P, F, 0;
            0, 0, 0, 0, 0, 0, 0
        );

        let port_pixels = hashmap!(
            (2, 1) => RefCell::new(UnionFindNode::new(1)),
            (1, 2) => RefCell::new(UnionFindNode::new(1)),

            (4, 1) => RefCell::new(UnionFindNode::new(2)),
            (5, 2) => RefCell::new(UnionFindNode::new(2)),

            (1, 4) => RefCell::new(UnionFindNode::new(3)),
            (2, 5) => RefCell::new(UnionFindNode::new(3)),

            (5, 4) => RefCell::new(UnionFindNode::new(4)),
            (4, 5) => RefCell::new(UnionFindNode::new(4)),
        );

        let (modified_skeleton_image, port_pixels) =
            unify_port_pixels(skeleton_image.clone(), port_pixels, 2).unwrap();

        assert_eq!(
            port_pixels.get(&(2, 1)).unwrap().borrow().clone_data(),
            port_pixels.get(&(1, 2)).unwrap().borrow().clone_data(),
        );

        assert_eq!(
            port_pixels.get(&(4, 1)).unwrap().borrow().clone_data(),
            port_pixels.get(&(5, 2)).unwrap().borrow().clone_data(),
        );

        assert_eq!(
            port_pixels.get(&(1, 4)).unwrap().borrow().clone_data(),
            port_pixels.get(&(2, 5)).unwrap().borrow().clone_data(),
        );

        assert_eq!(
            port_pixels.get(&(5, 4)).unwrap().borrow().clone_data(),
            port_pixels.get(&(4, 5)).unwrap().borrow().clone_data(),
        );

        assert_ne!(
            port_pixels.get(&(2, 1)).unwrap().borrow().clone_data(),
            port_pixels.get(&(4, 1)).unwrap().borrow().clone_data(),
        );

        assert_ne!(
            port_pixels.get(&(1, 2)).unwrap().borrow().clone_data(),
            port_pixels.get(&(1, 4)).unwrap().borrow().clone_data(),
        );

        assert_ne!(
            port_pixels.get(&(5, 2)).unwrap().borrow().clone_data(),
            port_pixels.get(&(5, 4)).unwrap().borrow().clone_data(),
        );

        assert_ne!(
            port_pixels.get(&(2, 5)).unwrap().borrow().clone_data(),
            port_pixels.get(&(4, 5)).unwrap().borrow().clone_data(),
        );

        assert_pixels_eq!(modified_skeleton_image, skeleton_image)
    }
    {
        let skeleton_image = gray_image!(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
            0, 0, 0, E, E, E, E, E, E, E, 0, 0, 0;
            0, 0, 0, P, 0, 0, 0, 0, 0, P, 0, 0, 0;
            0, E, P, F, P, E, 0, 0, 0, F, P, E, 0;
            0, E, 0, 0, 0, E, 0, 0, 0, P, 0, E, 0;
            0, E, 0, 0, 0, C, E, C, E, E, 0, E, 0;
            0, E, 0, 0, 0, E, 0, E, 0, 0, 0, E, 0;
            0, E, 0, E, E, C, E, C, 0, 0, 0, E, 0;
            0, E, 0, P, 0, 0, 0, E, 0, 0, 0, E, 0;
            0, E, P, F, 0, 0, 0, E, P, F, P, E, 0;
            0, 0, 0, P, 0, 0, 0, 0, 0, P, 0, 0, 0;
            0, 0, 0, E, E, E, E, E, E, E, 0, 0, 0;
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        );

        let port_pixels = hashmap!(
            (2, 3) => RefCell::new(UnionFindNode::new(1)),
            (4, 3) => RefCell::new(UnionFindNode::new(2)),
            (3, 2) => RefCell::new(UnionFindNode::new(3)),

            (9, 2) => RefCell::new(UnionFindNode::new(4)),
            (9, 4) => RefCell::new(UnionFindNode::new(5)),
            (10, 3) => RefCell::new(UnionFindNode::new(6)),

            (3, 8) => RefCell::new(UnionFindNode::new(7)),
            (3, 10) => RefCell::new(UnionFindNode::new(8)),
            (2, 9) => RefCell::new(UnionFindNode::new(9)),

            (8, 9) => RefCell::new(UnionFindNode::new(10)),
            (10, 9) => RefCell::new(UnionFindNode::new(11)),
            (9, 10) => RefCell::new(UnionFindNode::new(12)),
        );

        let (modified_skeleton_image, port_pixels) =
            unify_port_pixels(skeleton_image.clone(), port_pixels, 2).unwrap();

        assert_eq!(
            port_pixels.get(&(2, 3)).unwrap().borrow().clone_data(),
            port_pixels.get(&(4, 3)).unwrap().borrow().clone_data(),
        );
        assert_eq!(
            port_pixels.get(&(4, 3)).unwrap().borrow().clone_data(),
            port_pixels.get(&(3, 2)).unwrap().borrow().clone_data(),
        );
        assert_eq!(
            port_pixels.get(&(3, 2)).unwrap().borrow().clone_data(),
            port_pixels.get(&(5, 5)).unwrap().borrow().clone_data(),
        );

        assert_eq!(
            port_pixels.get(&(9, 2)).unwrap().borrow().clone_data(),
            port_pixels.get(&(9, 4)).unwrap().borrow().clone_data(),
        );
        assert_eq!(
            port_pixels.get(&(9, 4)).unwrap().borrow().clone_data(),
            port_pixels.get(&(10, 3)).unwrap().borrow().clone_data(),
        );
        assert_eq!(
            port_pixels.get(&(10, 3)).unwrap().borrow().clone_data(),
            port_pixels.get(&(7, 5)).unwrap().borrow().clone_data(),
        );

        assert_eq!(
            port_pixels.get(&(3, 8)).unwrap().borrow().clone_data(),
            port_pixels.get(&(3, 10)).unwrap().borrow().clone_data(),
        );
        assert_eq!(
            port_pixels.get(&(3, 10)).unwrap().borrow().clone_data(),
            port_pixels.get(&(2, 9)).unwrap().borrow().clone_data(),
        );
        assert_eq!(
            port_pixels.get(&(2, 9)).unwrap().borrow().clone_data(),
            port_pixels.get(&(5, 7)).unwrap().borrow().clone_data(),
        );

        assert_eq!(
            port_pixels.get(&(8, 9)).unwrap().borrow().clone_data(),
            port_pixels.get(&(10, 9)).unwrap().borrow().clone_data(),
        );
        assert_eq!(
            port_pixels.get(&(10, 9)).unwrap().borrow().clone_data(),
            port_pixels.get(&(9, 10)).unwrap().borrow().clone_data(),
        );
        assert_eq!(
            port_pixels.get(&(9, 10)).unwrap().borrow().clone_data(),
            port_pixels.get(&(7, 7)).unwrap().borrow().clone_data(),
        );

        let expected_skeleton_image = gray_image!(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
            0, 0, 0, E, E, E, E, E, E, E, 0, 0, 0;
            0, 0, 0, P, 0, 0, 0, 0, 0, P, 0, 0, 0;
            0, E, P, F, P, E, 0, 0, 0, F, P, E, 0;
            0, E, 0, 0, 0, E, 0, 0, 0, P, 0, E, 0;
            0, E, 0, 0, 0, P, E, P, E, E, 0, E, 0;
            0, E, 0, 0, 0, E, 0, E, 0, 0, 0, E, 0;
            0, E, 0, E, E, P, E, P, 0, 0, 0, E, 0;
            0, E, 0, P, 0, 0, 0, E, 0, 0, 0, E, 0;
            0, E, P, F, 0, 0, 0, E, P, F, P, E, 0;
            0, 0, 0, P, 0, 0, 0, 0, 0, P, 0, 0, 0;
            0, 0, 0, E, E, E, E, E, E, E, 0, 0, 0;
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        );

        assert_pixels_eq!(modified_skeleton_image, expected_skeleton_image)
    }
}

fn calculate_adjacency_map(
    skeleton_image: &GrayImage,
    port_pixels: &PortPixels,
) -> Result<AdjacencyMap, String> {
    let mut adjacency_list: AdjacencyMap = HashMap::new();
    let mut visited = HashSet::new();

    for (_, port_pixel_node) in port_pixels {
        let port_pixel_id = port_pixel_node.borrow().clone_data();

        if !adjacency_list.contains_key(&port_pixel_id) {
            adjacency_list.insert(port_pixel_id, RefCell::new(HashMap::new()));
        }
    }

    for (src_pixel, src_pixel_node) in port_pixels {
        visited.insert(*src_pixel);

        let src_pixel_id = src_pixel_node.borrow().clone_data();
        let mut src_adjacents = adjacency_list.get(&src_pixel_id).unwrap().borrow_mut();

        for port_pixel_neighbor in &four_neighbors(&skeleton_image, src_pixel.0, src_pixel.1) {
            if !(port_pixel_neighbor.1 == EDGE_PIXEL || port_pixel_neighbor.1 == PORT_PIXEL) {
                continue;
            }

            let mut current_pixel = port_pixel_neighbor.0;
            let mut prev_pixel = *src_pixel;

            loop {
                if skeleton_image.get_pixel(current_pixel.0, current_pixel.1)[0] == PORT_PIXEL {
                    if visited.contains(&current_pixel) {
                        break;
                    }

                    let dst_pixel_id = port_pixels
                        .get(&current_pixel)
                        .unwrap()
                        .borrow()
                        .clone_data();

                    if dst_pixel_id == src_pixel_id {
                        break;
                    }

                    let mut dst_adjacents = adjacency_list.get(&dst_pixel_id).unwrap().borrow_mut();

                    *(src_adjacents.entry(dst_pixel_id).or_insert(0)) += 1;
                    *(dst_adjacents.entry(src_pixel_id).or_insert(0)) += 1;

                    break;
                }

                let mut moved = false;

                for neighbor in &four_neighbors(&skeleton_image, current_pixel.0, current_pixel.1) {
                    if neighbor.0 != prev_pixel
                        && (neighbor.1 == EDGE_PIXEL || neighbor.1 == PORT_PIXEL)
                    {
                        prev_pixel = current_pixel;
                        current_pixel = neighbor.0;

                        moved = true;
                        break;
                    }
                }

                if !moved {
                    return Err(format!(
                        "calculate_adjacency_map: dead end edge on pixel: {:?}",
                        current_pixel
                    ));
                }
            }
        }
    }

    Ok(adjacency_list)
}

fn four_neighbors(image: &GrayImage, x: u32, y: u32) -> FourNeighbors {
    [
        ((x, y - 1), image.get_pixel(x, y - 1)[0]),
        ((x + 1, y), image.get_pixel(x + 1, y)[0]),
        ((x, y + 1), image.get_pixel(x, y + 1)[0]),
        ((x - 1, y), image.get_pixel(x - 1, y)[0]),
    ]
}

#[test]
fn calculate_adjacency_map_test() {
    const F: u8 = FOREGROUND_PIXEL;
    const P: u8 = PORT_PIXEL;
    const E: u8 = EDGE_PIXEL;

    {
        let skeleton_image = gray_image!(
            0, 0, 0, 0, 0, 0, 0;
            0, F, P, E, P, F, 0;
            0, P, 0, 0, 0, P, 0;
            0, E, 0, 0, 0, E, 0;
            0, P, 0, 0, 0, P, 0;
            0, F, P, E, P, F, 0;
            0, 0, 0, 0, 0, 0, 0
        );

        let port_pixels = hashmap!(
            (2, 1) => RefCell::new(UnionFindNode::new(1)),
            (1, 2) => RefCell::new(UnionFindNode::new(1)),
            (4, 1) => RefCell::new(UnionFindNode::new(2)),
            (5, 2) => RefCell::new(UnionFindNode::new(2)),
            (1, 4) => RefCell::new(UnionFindNode::new(3)),
            (2, 5) => RefCell::new(UnionFindNode::new(3)),
            (5, 4) => RefCell::new(UnionFindNode::new(4)),
            (4, 5) => RefCell::new(UnionFindNode::new(4)),
        );

        let expected_adjacency_map = hashmap!(
            1 => RefCell::new(hashmap!(
                2 => 1,
                3 => 1
            )),
            2 => RefCell::new(hashmap!(
                1 => 1,
                4 => 1,
            )),
            3 => RefCell::new(hashmap!(
                1 => 1,
                4 => 1,
            )),
            4 => RefCell::new(hashmap!(
                2 => 1,
                3 => 1,
            )),
        );

        assert_eq!(
            calculate_adjacency_map(&skeleton_image, &port_pixels).unwrap(),
            expected_adjacency_map
        )
    }
    {
        let skeleton_image = gray_image!(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
            0, F, F, F, P, E, P, F, F, F, 0;
            0, F, F, F, 0, 0, 0, F, F, F, 0;
            0, F, F, F, P, E, P, F, F, F, 0;
            0, P, 0, P, 0, 0, 0, P, 0, P, 0;
            0, E, 0, E, 0, 0, 0, E, 0, E, 0;
            0, P, 0, P, 0, 0, 0, P, 0, P, 0;
            0, F, F, F, P, E, P, F, F, F, 0;
            0, F, F, F, 0, 0, 0, F, F, F, 0;
            0, F, F, F, P, E, P, F, F, F, 0;
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        );

        let port_pixels = hashmap!(
            (4, 1) => RefCell::new(UnionFindNode::new(1)),
            (4, 3) => RefCell::new(UnionFindNode::new(1)),
            (1, 4) => RefCell::new(UnionFindNode::new(1)),
            (3, 4) => RefCell::new(UnionFindNode::new(1)),

            (6, 1) => RefCell::new(UnionFindNode::new(2)),
            (6, 3) => RefCell::new(UnionFindNode::new(2)),
            (7, 4) => RefCell::new(UnionFindNode::new(2)),
            (9, 4) => RefCell::new(UnionFindNode::new(2)),

            (1, 6) => RefCell::new(UnionFindNode::new(3)),
            (3, 6) => RefCell::new(UnionFindNode::new(3)),
            (4, 7) => RefCell::new(UnionFindNode::new(3)),
            (4, 9) => RefCell::new(UnionFindNode::new(3)),

            (7, 6) => RefCell::new(UnionFindNode::new(4)),
            (9, 6) => RefCell::new(UnionFindNode::new(4)),
            (6, 7) => RefCell::new(UnionFindNode::new(4)),
            (6, 9) => RefCell::new(UnionFindNode::new(4)),
        );

        let expected_adjacency_map = hashmap!(
            1 => RefCell::new(hashmap!(
                2 => 2,
                3 => 2,
            )),
            2 => RefCell::new(hashmap!(
                1 => 2,
                4 => 2,
            )),
            3 => RefCell::new(hashmap!(
                1 => 2,
                4 => 2,
            )),
            4 => RefCell::new(hashmap!(
                2 => 2,
                3 => 2,
            )),
        );

        assert_eq!(
            expected_adjacency_map,
            calculate_adjacency_map(&skeleton_image, &port_pixels).unwrap()
        )
    }
    {
        let skeleton_image = gray_image!(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
            0, 0, 0, E, E, E, E, E, E, E, 0, 0, 0;
            0, 0, 0, P, 0, 0, 0, 0, 0, P, 0, 0, 0;
            0, E, P, F, P, E, 0, 0, 0, F, P, E, 0;
            0, E, 0, 0, 0, E, 0, 0, 0, P, 0, E, 0;
            0, E, 0, 0, 0, P, E, P, E, E, 0, E, 0;
            0, E, 0, 0, 0, E, 0, E, 0, 0, 0, E, 0;
            0, E, 0, E, E, P, E, P, 0, 0, 0, E, 0;
            0, E, 0, P, 0, 0, 0, E, 0, 0, 0, E, 0;
            0, E, P, F, 0, 0, 0, E, P, F, P, E, 0;
            0, 0, 0, P, 0, 0, 0, 0, 0, P, 0, 0, 0;
            0, 0, 0, E, E, E, E, E, E, E, 0, 0, 0;
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        );

        let port_pixels = hashmap!(
            (2, 3) => RefCell::new(UnionFindNode::new(1)),
            (4, 3) => RefCell::new(UnionFindNode::new(1)),
            (3, 2) => RefCell::new(UnionFindNode::new(1)),
            (5, 5) => RefCell::new(UnionFindNode::new(1)),

            ( 9, 2) => RefCell::new(UnionFindNode::new(2)),
            ( 9, 4) => RefCell::new(UnionFindNode::new(2)),
            (10, 3) => RefCell::new(UnionFindNode::new(2)),
            ( 7, 5) => RefCell::new(UnionFindNode::new(2)),

            (3,  8) => RefCell::new(UnionFindNode::new(3)),
            (3, 10) => RefCell::new(UnionFindNode::new(3)),
            (2,  9) => RefCell::new(UnionFindNode::new(3)),
            (5,  7) => RefCell::new(UnionFindNode::new(3)),

            ( 8,  9) => RefCell::new(UnionFindNode::new(4)),
            (10,  9) => RefCell::new(UnionFindNode::new(4)),
            ( 9, 10) => RefCell::new(UnionFindNode::new(4)),
            ( 7,  7) => RefCell::new(UnionFindNode::new(4)),
        );

        let expected_adjacency_map = hashmap!(
            1 => RefCell::new(hashmap!(
                2 => 2,
                3 => 2,
            )),
            2 => RefCell::new(hashmap!(
                1 => 2,
                4 => 2,
            )),
            3 => RefCell::new(hashmap!(
                1 => 2,
                4 => 2,
            )),
            4 => RefCell::new(hashmap!(
                2 => 2,
                3 => 2,
            )),
        );

        assert_eq!(
            calculate_adjacency_map(&skeleton_image, &port_pixels).unwrap(),
            expected_adjacency_map
        )
    }
}

fn extract_bits(adjacency_map: &AdjacencyMap) -> Result<Vec<u8>, String> {
    let mut visited = hashset!();
    let mut density = 0;
    let mut bits = vec![];
    let mut layer_index = 1;

    let (center_vertex, mut base_vertex, mut direction_vertex) =
        find_marker_vertices(adjacency_map)?;
    visited.insert(center_vertex);

    loop {
        let (frame_vertices, bridge_vertex) = get_frame_vertices(
            StartVertex::Bridge,
            &base_vertex,
            &direction_vertex,
            &adjacency_map,
            &visited,
        );

        if density == 0 {
            density = frame_vertices.len() + 1;
        }

        match bridge_vertex {
            None => {
                break;
            }

            Some(bridge_vertex) => {
                visited.insert(bridge_vertex);
                visited.extend(&frame_vertices);

                let next_base_vertex =
                    get_next_vertex(&bridge_vertex, &2, &adjacency_map, &visited, &hashset!())
                        .unwrap();
                let next_direction_vertex =
                    get_next_vertex(&bridge_vertex, &1, &adjacency_map, &visited, &hashset!())
                        .unwrap();

                let (next_frame_vertices, _) = get_frame_vertices(
                    StartVertex::Base,
                    &next_base_vertex,
                    &next_direction_vertex,
                    &adjacency_map,
                    &visited,
                );

                for (i, frame_vertex) in frame_vertices.iter().enumerate() {
                    let start_index = i + i / layer_index;
                    let num_edges = if (i + 1) % layer_index == 0 { 3 } else { 2 };

                    for i in 0..num_edges {
                        match adjacency_map
                            .get(frame_vertex)
                            .unwrap()
                            .borrow()
                            .get(&next_frame_vertices[start_index + i])
                        {
                            Some(1) => {
                                bits.push(0xFF);
                            }

                            _ => {
                                bits.push(0);
                            }
                        }
                    }
                }

                layer_index += 1;
                base_vertex = next_base_vertex;
                direction_vertex = next_direction_vertex;
            }
        }
    }

    Ok(bits)
}

fn find_marker_vertices(
    adjacency_map: &AdjacencyMap,
) -> Result<(VertexId, VertexId, VertexId), String> {
    for (src, adjacents) in adjacency_map {
        let adjacents = adjacents.borrow();

        if adjacents.len() != 2 {
            continue;
        }

        for (dst, weight) in &*adjacents {
            if *weight != 2 {
                continue;
            }

            for (direction_vertex, weight) in &*adjacency_map.get(dst).unwrap().borrow() {
                if *weight != 2 {
                    continue;
                }

                match adjacents.get(direction_vertex) {
                    Some(1) => return Ok((*src, *dst, *direction_vertex)),
                    _ => {}
                }
            }
        }
    }

    Err("invalid MC code: cannot find center vertex".to_string())
}

#[derive(PartialEq)]
enum StartVertex {
    Base,
    Bridge,
}

fn get_frame_vertices(
    start_vertex: StartVertex,
    base_vertex: &VertexId,
    direction_vertex: &VertexId,
    adjacency_map: &AdjacencyMap,
    visited: &HashSet<VertexId>,
) -> (Vec<VertexId>, Option<VertexId>) {
    let mut frame_visited = hashset!();
    let mut current_vertex = *direction_vertex;
    let mut vertices = vec![];
    let mut vertices_reverse = vec![];
    let mut bridge_vertex = None;

    frame_visited.insert(*base_vertex);

    loop {
        frame_visited.insert(current_vertex);

        if is_bridge_vertex(&current_vertex, &adjacency_map, visited) {
            bridge_vertex = Some(current_vertex);

            if start_vertex == StartVertex::Base {
                vertices.push(current_vertex);
            }

            break;
        }

        vertices.push(current_vertex);

        match get_next_vertex(&current_vertex, &2, &adjacency_map, visited, &frame_visited) {
            Some(next_vertex) => {
                current_vertex = next_vertex;
            }

            None => {
                break;
            }
        }
    }

    match bridge_vertex {
        None => {}
        Some(_) => {
            current_vertex = *base_vertex;

            loop {
                frame_visited.insert(current_vertex);

                if start_vertex == StartVertex::Bridge || current_vertex != *base_vertex {
                    vertices_reverse.push(current_vertex);
                }

                match get_next_vertex(&current_vertex, &2, &adjacency_map, visited, &frame_visited)
                {
                    Some(next_vertex) => {
                        current_vertex = next_vertex;
                    }
                    None => {
                        break;
                    }
                }
            }
        }
    }

    vertices_reverse.reverse();

    match start_vertex {
        StartVertex::Base => {
            vertices.extend(vertices_reverse);
            (vertices, bridge_vertex)
        }

        StartVertex::Bridge => {
            vertices_reverse.extend(vertices);
            (vertices_reverse, bridge_vertex)
        }
    }
}

#[test]
fn get_frame_vertices_test() {
    {
        let adjacency_map = hashmap!(
            1 => RefCell::new(hashmap!(
                2 => 2,
                3 => 1,
            )),
            2 => RefCell::new(hashmap!(
                1 => 2,
                3 => 2,
                4 => 2,
            )),
            3 => RefCell::new(hashmap!(
                2 => 2,
                4 => 2,
            )),
            4 => RefCell::new(hashmap!(
                2 => 2,
                3 => 2,
                5 => 2,
                6 => 1,
            )),
            5 => RefCell::new(hashmap!(
                4 => 2,
                6 => 2,
                10 => 2,
            )),
            6 => RefCell::new(hashmap!(
                5 => 2,
                7 => 2,
            )),
            7 => RefCell::new(hashmap!(
                6 => 2,
                8 => 2,
            )),
            8 => RefCell::new(hashmap!(
                7 => 2,
                9 => 2,
            )),
            9 => RefCell::new(hashmap!(
                8 => 2,
                10 => 2,
                11 => 2,
                12 => 1,
            )),
            10 => RefCell::new(hashmap!(
                9 => 2,
                5 => 2,
            )),
            11 => RefCell::new(hashmap!(
                9 => 2,
                12 => 2,
                19 => 2,
            )),
            12 => RefCell::new(hashmap!(
                9 => 1,
                11 => 2,
                13 => 2,
            )),
            13 => RefCell::new(hashmap!(
                12 => 2,
                14 => 2,
            )),
            14 => RefCell::new(hashmap!(
                13 => 2,
                15 => 2,
            )),
            15 => RefCell::new(hashmap!(
                14 => 2,
                16 => 2,
            )),
            16 => RefCell::new(hashmap!(
                15 => 2,
                17 => 2,
            )),
            17 => RefCell::new(hashmap!(
                16 => 2,
                18 => 2,
            )),
            18 => RefCell::new(hashmap!(
                17 => 2,
                19 => 2,
            )),
            19 => RefCell::new(hashmap!(
                18 => 2,
                11 => 2,
            )),
        );

        let mut visited = hashset!(1);

        assert_eq!(
            (vec![2, 3], Some(4)),
            get_frame_vertices(StartVertex::Bridge, &2, &3, &adjacency_map, &visited)
        );

        visited = hashset!(1, 2, 3, 4);

        assert_eq!(
            (vec![6, 7, 8, 9, 10], Some(9)),
            get_frame_vertices(StartVertex::Base, &5, &6, &adjacency_map, &visited)
        );

        assert_eq!(
            (vec![10, 5, 6, 7, 8], Some(9)),
            get_frame_vertices(StartVertex::Bridge, &5, &6, &adjacency_map, &visited)
        );

        visited = hashset!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        assert_eq!(
            (vec![12, 13, 14, 15, 16, 17, 18, 19], None),
            get_frame_vertices(StartVertex::Base, &11, &12, &adjacency_map, &visited)
        );
    }
    {
        let adjacency_map = hashmap!(
            1 => RefCell::new(hashmap!(
                2 => 2,
                3 => 1,
            )),
            2 => RefCell::new(hashmap!(
                1 => 2,
                3 => 2,
                4 => 2,
            )),
            3 => RefCell::new(hashmap!(
                2 => 2,
                4 => 2,
                5 => 2,
                6 => 1,
            )),
            4 => RefCell::new(hashmap!(
                2 => 2,
                3 => 2,
            )),
            5 => RefCell::new(hashmap!(
                3 => 2,
                6 => 2,
                10 => 2,
            )),
            6 => RefCell::new(hashmap!(
                3 => 1,
                5 => 2,
                7 => 2,
            )),
            7 => RefCell::new(hashmap!(
                6 => 2,
                8 => 2,
            )),
            8 => RefCell::new(hashmap!(
                7 => 2,
                9 => 2,
            )),
            9 => RefCell::new(hashmap!(
                8 => 2,
                10 => 2,
            )),
            10 => RefCell::new(hashmap!(
                9 => 2,
                5 => 2,
            )),
        );

        let mut visited = hashset!(1);

        assert_eq!(
            (vec![4, 2], Some(3)),
            get_frame_vertices(StartVertex::Bridge, &2, &3, &adjacency_map, &visited)
        );

        visited = hashset!(1, 2, 3, 4);

        assert_eq!(
            (vec![6, 7, 8, 9, 10], None),
            get_frame_vertices(StartVertex::Base, &5, &6, &adjacency_map, &visited)
        );
    }
}

fn is_bridge_vertex(
    current: &VertexId,
    adjacency_map: &AdjacencyMap,
    visited: &HashSet<VertexId>,
) -> bool {
    let mut count = 0;

    for (next_vertex, weight) in &*adjacency_map.get(current).unwrap().borrow() {
        if *weight != 2 {
            continue;
        }

        if visited.contains(&next_vertex) {
            continue;
        }

        count += 1
    }

    count == 3
}

fn get_next_vertex(
    current: &VertexId,
    weight: &Weight,
    adjacency_map: &AdjacencyMap,
    visited: &HashSet<VertexId>,
    frame_visited: &HashSet<VertexId>,
) -> Option<VertexId> {
    for (next_vertex, next_weight) in &*adjacency_map.get(current).unwrap().borrow() {
        if next_weight != weight {
            continue;
        }

        if visited.contains(&next_vertex) {
            continue;
        }

        if frame_visited.contains(&next_vertex) {
            continue;
        }

        return Some(*next_vertex);
    }

    None
}

fn decode_bits(bits: Vec<u8>) -> String {
    let mut bits = bits;
    let mut bytes = vec![];

    match decode_four_bits(&bits[0..4]) {
        0 => match demask(&mut bits[4..]) {
            Ok(data_bits) => {
                for chunk in data_bits.chunks(8) {
                    if chunk.len() < 8 {
                        break;
                    }

                    let byte = decode_eight_bits(chunk);

                    if byte == 0 {
                        break;
                    }

                    bytes.push(byte);
                }
            }

            Err(error) => return error,
        },

        n => return format!("unsupported MC code version {:?}", n),
    }

    match String::from_utf8(bytes) {
        Ok(string) => string,
        Err(error) => format!("invalid UTF8 string. error: {:?}", error),
    }
}

fn demask(bits: &mut [u8]) -> Result<&[u8], String> {
    match decode_four_bits(&bits[0..4]) {
        0 => Ok(&bits[4..]),

        1 => {
            for (i, bit) in bits[4..].iter_mut().enumerate() {
                if i % 2 != 0 {
                    *bit = !*bit;
                }
            }

            Ok(&bits[4..])
        }

        n => Err(format!("unsupported MC code mask pattern {:?}", n)),
    }
}

fn decode_four_bits(bits: &[u8]) -> u8 {
    (bits[0] & 0b1000) | (bits[1] & 0b0100) | (bits[2] & 0b0010) | (bits[3] & 0b0001)
}

fn decode_eight_bits(bits: &[u8]) -> u8 {
    (bits[0] & 0b10000000)
        | (bits[1] & 0b01000000)
        | (bits[2] & 0b00100000)
        | (bits[3] & 0b00010000)
        | (bits[4] & 0b00001000)
        | (bits[5] & 0b00000100)
        | (bits[6] & 0b00000010)
        | (bits[7] & 0b00000001)
}
