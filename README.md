# YOLOv8 Escalator People Counting System

This project utilizes YOLOv8 for real-time object detection and SORT (Simple Online and Realtime Tracking) algorithm for tracking individuals on an escalator. The system accurately counts the number of people moving up and down the escalator separately.
![People Counting Demo](demo.gif)


## Installation

1. Clone the repository:

    ```
    git clone https://github.com/your_username/escalator-people-counter.git
    ```

2. Install the required libraries:

    ```
    pip install -r requirements.txt
    ```

3. Download the YOLOv8 weights file (`yolov8l.pt`) and place it in the `Yolo-Weights` directory.

## Usage

1. Replace the video file `people.mp4` in the `Videos` directory with your escalator footage.

2. Run the Python script:

    ```
    python escalator_people_counter.py
    ```

3. The script will process the video frames, perform object detection using YOLOv8, and track individuals on the escalator.

4. The output frames with bounding boxes and counts will be saved in the `output_frames` directory.

5. Once the processing is complete, the script will create a new video (`output_video.mp4`) showing the escalator with people counted separately moving up and down.

## Output Video

Here is the output video demonstrating the people counting on the escalator:

<video width="640" height="480" controls>
  <source src="output_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Algorithm

1. **Object Detection**: YOLOv8 detects people on the escalator with high accuracy.

2. **Object Tracking**: The SORT algorithm tracks individuals across frames to maintain consistency.

3. **Counting**: Individuals crossing predefined regions on the escalator are counted separately for up and down directions.

## Customization

- Adjust the confidence threshold (`conf > 0.3`) in the code to control the detection sensitivity.
- Modify the region limits (`limitsUP` and `limitsDown`) to fit the specific escalator layout.

## Dependencies

- [YOLOv8](https://github.com/ultralytics/yolov5) for object detection.
- [SORT](https://github.com/abewley/sort) for object tracking.
- [cvzone](https://github.com/cvzone/cvzone) for drawing bounding boxes and text on images.

## Author

- [Joy Karmoker](https://github.com/JoyKarmoker)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

