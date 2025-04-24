import cv2
import numpy as np
from configYOLO import classNames, COLORS, FONT, FONT_SCALE, THICKNESS, load_yolo_model
from configFastSAM import load_sam_model, get_silhouette
import math
from inputFromCamera import InputFromCamera


def main():
    # initialize input frow webvam or dataset
    input_source = InputFromCamera(use_webcam=True, dataset_path=None)  # Change use_webcam as needed
    model = load_yolo_model()
    sam_predictor = load_sam_model()

    frame_count = 0
    alpha = 0.4
    mask_color = (0, 255, 0)  # green

    while True:
        try:
            print("___________________________________")
            print("Waiting for frame...")
            # get next frame (rgbd)
            rgb_frame, depth_frame = input_source.get_frame()

            frame_count += 1
            results = model(rgb_frame, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # calc confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    # continue only if confidence>50%
                    if confidence < 0.5:
                        print(f"Skipping detection with confidence {confidence:.2f}")
                        continue

                    # find class name
                    cls = int(box.cls[0])
                    class_name = classNames[cls]
                    print("Class name -->", class_name)

                    # draw YOLO bounding box
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # put class name
                    org = [x1, y1]
                    cv2.putText(rgb_frame, class_name, org, FONT, FONT_SCALE, COLORS[cls], THICKNESS)

                    # process mask only every 1 frames (an sas kollaei poly anevaste to kathe pote tha fortwnei to fastSAM)
                    if frame_count % 1 == 0:
                        try:
                            mask = get_silhouette(sam_predictor, rgb_frame, [x1, y1, x2, y2])
                            print("Mask sum:", np.sum(mask))  

                            if np.sum(mask) == 0:
                                print("Empty mask")
                                continue

                            # colored mask
                            overlay = rgb_frame.copy()
                            overlay[mask] = mask_color

                            # place mask on frame
                            rgb_frame = cv2.addWeighted(overlay, alpha, rgb_frame, 1 - alpha, 0)

                        except Exception as e:
                            print(f"Error during segmentation: {e}")

            cv2.imshow("Object Localization and Segmentation", rgb_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            break

    input_source.release()  
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
