# Object Transfer Analysis

This project aims to track people and objects in video, detect object handovers,
and reconstruct each object's history. The intended pipeline combines object
detection and tracking with temporal possession and transfer rules. A learned
classifier and dashboard are later extensions.

The project is at the **pilot detection and tracking** stage. Transfer
inference, possession, history, and the dashboard are not implemented yet.

## Step 1: Prepare a small pilot

Record 5-10 short videos using a fixed camera, or use suitable existing videos
with permission. Aim for 10-30 seconds each, with two people and one clearly
visible object, such as a cup, bottle, or book. Film hands and objects clearly,
and leave a few seconds before and after each interaction.

Include both handovers and non-handover examples:

- A gives an object to B.
- B gives it back to A, or the same object changes hands twice.
- A holds or moves an object without giving it away.
- A places an object on a table, or B reaches near it without taking it.

Start with good lighting, a simple background, and limited occlusion. Add harder
examples after the basic pipeline works. Avoid including private information in
the frame, and obtain consent from everyone recorded. Keep the original videos
private and share them only with permission.

Place the videos under `data/raw_videos/`, for example
`data/raw_videos/pilot_001.mp4`. Raw videos are excluded by `.gitignore`.
Do not upload them to a public repository without permission.

## Step 2: Label the pilot

### Review handovers locally

Start the local review tool from the project directory:

```bash
source .venv/bin/activate
python -m src.annotation.review
```

Open the **complete URL printed in the terminal**, including the token after
`#`. Select a video and use the slider or frame buttons to inspect the original
frames. Enter anonymous IDs that remain consistent within the video, then mark
the start and end frames using the buttons. Mark the first frame where the
receiver clearly controls the object if it is visible; otherwise leave the
transfer frame blank. Save each real event once. Use a distinct event ID per
event in a video. If there is no handover, do not add a row.

The tool appends saved events to `data/annotations/transfers.csv` and does not
modify the videos. It does not record negative-review status, person identity
descriptions, or bounding boxes. Keep a separate private record of which
anonymous IDs refer to which person in each video, and annotate boxes separately.
To correct a mistake, stop the tool and edit the CSV manually; do not remove its
header. Do not copy model track IDs without reviewing them. The tool binds only
to this computer, but anyone with local access can view the footage: do not
share the URL. Stop it with Ctrl+C when finished.

`data/annotations/videos.csv` registers the five pilot videos as one recording
session. They are all assigned to the training split to avoid splitting one
session across train and test. The annotation CSVs for frames, boxes and
transfers contain headers only. Fill them in after reviewing the videos.
Use UTF-8 CSV with a header, one row per item, and consistent IDs. Do not put
real participant names in the files. Frame indices are **zero-based**: the
first decoded frame is 0.

1. **Register videos:** Add one row per video to `data/annotations/videos.csv`.
   Give each recording session an anonymous `session_id`. Record frame rate,
   width, height, and frame count from the video metadata. Existing entries use
   the nominal video-track frame rate, OpenCV-decoded frame count, and display
   dimensions after applying the video's orientation. Use the same decoder and
   image orientation for annotation and tracking.
2. **Choose splits:** Add each video once to `data/splits/videos.csv`, using
   `train`, `val`, or `test`. Keep all clips from a recording session in the
   same split; where possible, also keep participants separate. With a tiny
   pilot, test results are preliminary. Record separate sessions for validation
   and testing; until then, there is no held-out evaluation. Do not tune rules
   on the test set.
3. **Label transfers:** Add one row per actual handover to
   `data/annotations/transfers.csv`. Give each person and object a stable ID
   within the video. Define the event window from the start of the handover
   attempt through the receiver gaining control. `transfer_frame` is the frame
   where the receiver first clearly has control, if identifiable; otherwise
   leave it empty. Use inclusive start and end frames. Videos with no handover
   have no transfer rows; they remain listed in the video and split files.
4. **Label sampled boxes:** Add each reviewed frame to
   `data/annotations/reviewed_frames.csv`, then add its visible people and
   relevant objects to `data/annotations/boxes.csv`. Start with a manageable
   sample (for example one frame per second), plus frames before, during, and
   after each handover. Review **all** visible people and in-scope objects in
   each listed frame; an unreviewed frame is not a negative example. Add rows
   for visible, partially occluded instances as well, using `occluded` = `1`;
   use `0` otherwise. Do not create boxes for completely invisible instances.

Boxes use absolute pixel coordinates `x_min,y_min,x_max,y_max` in the original
video frame, with the origin at the top left. The maximum coordinates are
exclusive; require `0 <= x_min < x_max <= width` and
`0 <= y_min < y_max <= height`. Use `person` for people and a consistent
lowercase class such as `cup`, `bottle`, or `book` for objects. Keep `entity_id`
stable throughout a video, including across occlusion, and use the same IDs in
the transfer file. Distinct objects of the same class get different IDs.

Example (illustrative only):

```csv
video_id,event_id,object_id,giver_id,receiver_id,start_frame,end_frame,transfer_frame
pilot_001,event_01,object_01,person_01,person_02,120,150,145
```

Bounding boxes and identities are stored separately from event windows. Standard
YOLO detection labels do not encode persistent IDs or transfer events. A future
exporter can convert these boxes to normalized YOLO labels after checking class
mapping and image sizes; the CSV files are not directly usable as YOLO labels.
Frame timestamps should be obtained from the video decoder. For constant-frame-
rate video, frame index divided by frames per second is an approximation in
seconds; variable-frame-rate footage needs its actual frame timestamps.

### Label boxes (for detector fine-tuning)

Pretrained YOLO finds the transferred object in too few handover frames (see
below), so the next step is to label boxes and fine-tune. Start the box tool
with the larger model's predictions as **unverified** suggestions:

```bash
python -m src.annotation.boxes --prelabels results/tracking_yolo11x_1280_c005
```

Open the printed URL (including the token). The tool suggests every 5th frame
plus each labeled event's start, end and transfer frames. For each frame, check
every box, delete wrong ones, draw missing ones (drag on the image), give every
box an entity ID (`p1`-`p3` for people, the video's object ID such as `b1` or `l1`
for the object; IDs are prefilled from nearby saved frames or the event labels,
so check them), mark partially hidden items as occluded, then press **Save frame**.
Only saved frames are written to `data/annotations/boxes.csv` and
`reviewed_frames.csv`; suggestions alone are never recorded. Re-saving a frame
replaces its boxes. A saved frame without a bottle or laptop box is later used
as a "no object here" example, so label the object whenever it is visible, even
if small or held in a hand; use **Next saved frame with no object box** to review
these. Do not share the URL; stop with Ctrl+C.

### Fine-tune the detector

```bash
python -m src.detection.export_yolo          # reviewed frames -> data/yolo (split per video)
python -m src.detection.train                # fine-tune YOLO11s (about 20 min on Apple GPU)
python -m src.pipeline data/raw_videos/*.MOV --weights runs/pilot_yolo11s/weights/best.pt \
  --image-size 1280 --confidence 0.15 --output-dir results/tracking_finetuned
python -m experiments.detection_coverage results/tracking_finetuned
```

Splits are per video (`data/splits/videos.csv`): pilot_001-003 train, pilot_004-005
validation. All videos come from one session, so these are optimistic estimates,
not an independent test. Pilot result (60-epoch budget, early-stopped at 54):

| Held-out video | Pretrained YOLO11x (conf 0.05) | Fine-tuned (conf 0.15) |
|---|---|---|
| pilot_004 laptop, event frames with a laptop prediction | 5 of 92 | 88 of 92 |
| pilot_005 bottle, event frames with a bottle prediction | 27 of 97 | 57 of 97 |

Held-out mAP50 on validation frames: person 0.995, laptop 0.74, bottle 0.33.
Known problems: pilot_004 has 70 frames with a false bottle prediction, pilot_001 has
24 frames with duplicate bottle predictions, and bottle recall on pilot_005 is low.
Coverage counts any prediction of the right class and does not check its location.

### Detect transfers and rebuild object histories

```bash
python -m src.interaction.hand_detection data/raw_videos/*.MOV   # people + wrist keypoints
python -m src.transfer.analyze                                    # -> results/events.csv
python -m experiments.transfer_evaluation                         # score against labels
```

Method: a YOLO11 pose model tracks people and their wrists (a proxy for hands). In each
frame the person whose wrist is nearest the detected object's centre (within 0.12 person
heights) is its holder; frames where the object is missing, no wrist is close, or two
people tie count as unclear. A state machine (`src/transfer/state_machine.py`) reports a
transfer when a different person holds the object for 8 frames (gaps up to 4 tolerated).
`src/history/event_store.py` stores events and rebuilds each object's holder history.

Evaluation: a predicted event is correct if the object, giver and receiver match and its
transfer frame is within the labeled window plus 15 frames. Person IDs are mapped to your
labels by box overlap (100% vote purity on every track). Defaults were chosen by grid
search on the training videos only:

| Videos | Labeled events | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| train (pilot_001-003) | 7 | 4 | 4 | 3 | 0.50 | 0.57 | 0.53 |
| validation (pilot_004-005) | 4 | 3 | 0 | 1 | 1.00 | 0.75 | 0.86 |

The validation set is only 4 events from one session, so treat its score as anecdotal.
Known failure: when the object is passed *across* a third person standing between the
giver and receiver (pilot_002, pilot_003), the middle person's wrists are near the object
in the image, so they are wrongly reported as a holder. Fixing this needs depth or
arm-direction cues, or a learned classifier trained on more footage. Detection gaps also
matter: pilot_005's bottle is found in only 62 of 117 frames. A Random Forest classifier
(`src/transfer/classifier.py`) and the dashboard are not implemented.

## Next development milestones

1. Inspect the pilot videos and confirm annotations and train/validation/test
   splits.
2. Run a pretrained person/object detector and tracker; check whether selected
   object classes are detected reliably before considering fine-tuning.
3. Add hand/contact and possession features, then a rule-based transfer state
   machine. Track uncertainty and do not assume every nearby object is held.
4. Evaluate detection/tracking and transfer events on held-out videos, including
   false alarms on negative examples. Add an optional classifier only if enough
   labeled examples are available.
5. Add object histories and a dashboard after the baseline works end to end.

## Run the pilot detector and tracker

The pilot was run with the existing Python 3.14 environment, Ultralytics
8.4.211, OpenCV 5.0.0, PyTorch 2.15.0, and lap 0.5.14. For a fresh
environment, install only the dependencies needed for this stage:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install ultralytics lap
python -m src.pipeline data/raw_videos/*.MOV
```

The default `yolo11n.pt` COCO model downloads on first use; inference runs
locally. Choose the model and optional COCO class IDs if needed:

```bash
python -m src.pipeline data/raw_videos/IMG_5223.MOV \
  --output-dir results/person-cup --classes 0 41 --confidence 0.25
```

For small objects, try a larger YOLO11 model and input resolution. This uses
more computation and may still miss or misclassify objects:

```bash
python -m src.pipeline data/raw_videos/*.MOV \
  --weights yolo11s.pt --image-size 1280 \
  --output-dir results/tracking_yolo11s_1280
```

The standard COCO class IDs for person, bottle and laptop are 0, 39 and 63.
You may use `--classes 0 39 63` to hide other detections when reviewing, but
filtering classes does not make the model more accurate. Lowering `--confidence`
can reveal weaker predictions and may also increase false positives; ByteTrack
has its own thresholds, so not every low-confidence prediction gets a track.
Neither option guarantees detection. Review against manually labeled frames.

Each input produces an annotated MP4 and a CSV in `results/tracking/` by default.
CSV rows contain the zero-based frame number, approximate time, track ID, class,
confidence, and pixel coordinates. A blank track ID means the detection was not
assigned an ID. IDs are local to each video and may change or be wrong after
occlusion. Outputs have no audio and use a constant nominal frame rate; their
timestamps may not match variable-frame-rate source footage exactly.

The pretrained detector only recognizes its COCO classes, and a class name is
a prediction, not verified ground truth. Review the videos and compare against
manual labels before relying on them. Detection and tracking do not establish
possession or handover. The pipeline refuses to overwrite output files; use a
new output directory for another run, or explicitly remove files from a failed
run before retrying. Raw footage and generated results are ignored by Git; do
not share them without participant consent.

The full `requirements.txt` remains a preliminary, unpinned list for later
stages and has not been validated as a complete environment.

Run the focused tests with:

```bash
python -m unittest tests.test_pipeline -v
```

## Check object detection against labeled transfers

```bash
python -m experiments.detection_coverage results/tracking_yolo11s_1280
```

This reports, for each labeled event, the share of frames in the event window
that contain a prediction of the transferred object's class (`b*` object IDs are
treated as bottles and `l*` as laptops). It is only a necessary condition: a
prediction may still be the wrong object or location, so it is not a detector
accuracy measure. On the pilot, pretrained YOLO11s at 1280 pixels covered 6.8%
of event frames, and YOLO11x with `--confidence 0.05` covered 29.4%. Pretrained
COCO weights are not reliable enough here; fine-tuning on labeled boxes is the
next step.
