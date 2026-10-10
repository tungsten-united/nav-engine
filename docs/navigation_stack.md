<!-- From tungsten-united/demo-repository@c5dc26f (branch navigation), by LPA. Kept verbatim; the pipeline it describes is now nav/ (see README). -->

# Indoor Navigation Stack — Development Plan

**Project:** AI-Assisted Indoor Navigation for Visually Impaired People  
**Platform:** Android smartphone, accessed through a browser via HTTPS  
**Navigation approach:** Visual Place Recognition (VPR) + Orientation-Aware Semantic Graph + Sensor-Based Turn Detection  
**Demo environment:** Hackathon venue — Entrance, Stage, Kitchen, Drinks Area  
**Objective:** Demonstrate an end-to-end navigation experience within 60 seconds.

---

## 1. System Overview

The system consists of two connected phases.

### Phase A — Mapping (Team Integration)

A teammate records a walkthrough of the environment using a smartphone. The video is processed to extract:

- Important locations and semantic landmarks.
- Intermediate navigation checkpoints, particularly at intersections and turns.
- Traversable connections between checkpoints.
- Spatial relationships between visible landmarks.
- Approach directions and departure directions.
- Reference images for subsequent visual localization.

A VLM helps interpret the recorded environment and generates a preliminary orientation-aware semantic graph.

The graph is reviewed before navigation. The VLM may propose spatial relations, but it must not independently establish unobserved connections, safe passage, or precise directions.

### Phase B — Navigation (Our Responsibility)

The Android webapp uses its camera to recognize the user's surroundings and locate them within the generated map.

A navigation controller uses the graph and phone orientation measurements to determine which waypoint to reach next and what guidance to provide.

The system delivers voice instructions and haptic feedback through browser APIs.

**Main pipeline:**

Mapping video → Semantic-spatial map → Live camera → Visual localization → Orientation estimation → Graph navigation → Voice and haptic feedback.

---

## 2. Interface Between Mapping and Navigation

Before receiving the video, we need to agree with the mapping team on the structure of the generated map.

The mapping pipeline must produce four categories of information.

### 2.1 Navigation nodes

Each node represents a recognizable location or checkpoint.

Required information:

- Unique node identifier.
- Semantic label, such as Entrance, Stage or Drinks.
- Node type: destination, corridor, intersection, doorway or checkpoint.
- Reference images.
- Optional semantic description.
- Verification status.

The map should include intermediate checkpoints, not just the four final destinations. An intersection may be more important for navigation than a recognizable room.

### 2.2 Navigation edges

Each edge represents a physically traversed and verified connection between two checkpoints.

Required information:

- Starting node and destination node.
- Whether the connection has been verified.
- Expected next checkpoint.
- Traversal direction.
- Applicable navigation instruction.
- Reference views associated with that transition.

Edges should be directional. Reverse travel must be verified separately rather than assumed.

### 2.3 Spatial and directional relationships

The VLM should extract spatial relationships from video sequences.

For example, when approaching from the entrance:

- The drinks station appears on the right.
- The corridor continues straight ahead.
- The stage is visible further along the corridor.

These relations must be stored relative to a specific recorded viewpoint and orientation.

This allows the navigation system to update directional relationships when the user rotates.

For example:

- Initially: Corridor ahead, Drinks right.
- After a verified 90° right rotation: Drinks ahead, Corridor left.

The transformation can be calculated geometrically without making another VLM request, provided the user remains at the same checkpoint.

The map should distinguish **observed**, **inferred**, and **verified** relationships.

### 2.4 Reference images

For each checkpoint, store multiple images captured from useful directions:

- Images approaching the checkpoint.
- Images near the checkpoint itself.
- Images facing each relevant outgoing path.

These images will be used during live navigation for both localization and orientation verification.

---

## 3. Initial Map Generation

**Dependency:** The teammate's recorded video.

Once the recording is available, we process it to construct our first usable navigation map.

### Step 1 — Extract keyframes

Extract representative frames along the recorded walking path, with additional frames around intersections, turns and important landmarks.

Avoid keeping unnecessary near-duplicate images.

### Step 2 — Semantic analysis

Use a VLM to identify landmarks and propose spatial relationships.

The VLM should describe what is visible and distinguish observations from uncertain inferences.

### Step 3 — Construct the graph

Create nodes for destinations and navigation checkpoints.

Create candidate edges following the path actually traversed during recording.

Do not create edges merely because two places appear close together in an image.

### Step 4 — Establish directional information

Associate incoming and outgoing views with each checkpoint.

Estimate or annotate relative directions such as left, right and straight. Validate any proposed turn against the actual recorded movement.

If the video does not preserve sufficient orientation evidence, the mapper must annotate the direction manually.

### Step 5 — Verify and finalize

Inspect the graph, correct landmark labels, confirm connections, and validate turning instructions.

The first deliverable is a **verified, orientation-aware navigation graph with corresponding reference images**.

---

## 4. Visual Localization

**Goal:** Estimate where the user is within the mapped environment using live smartphone images.

### 4.1 Visual Place Recognition

Use a pretrained visual encoder to convert camera frames into embedding vectors.

Candidate models:

- DINOv2 — general visual feature representations.
- CLIP — accessible baseline with semantic image representations.
- A specialized VPR model — potential later improvement.

For the initial demo, start with one pretrained model and evaluate its recognition performance on actual venue images.

During mapping, generate and save embeddings for reference images.

During navigation, generate an embedding from each sampled camera frame and compare it against stored references.

The output should include:

- Most likely checkpoint candidates.
- Similarity scores.
- Associated viewing direction, where identifiable.
- Observation timestamp.

### 4.2 Temporal localization

A single camera frame should not determine the user's location.

Combine observations across multiple frames to identify stable matches.

The localization estimator should consider:

- The strongest visual match.
- The difference between competing matches.
- Agreement across consecutive observations.
- The previously confirmed location.
- Plausibility within the navigation graph.

The output is a location estimate and uncertainty status, not a guaranteed physical position.

### 4.3 Checkpoint arrival

Recognizing a landmark from a distance does not mean the user has reached its checkpoint.

For the MVP, use reference images captured from an established checkpoint position. Require additional evidence of proximity or explicit checkpoint confirmation before advancing.

We will not attempt centimeter-level localization or continuous metric position estimation.

---

## 5. User Orientation Estimation

**Goal:** Determine the direction the phone is facing relative to the recorded navigation graph.

We will combine relative phone rotation with visual orientation verification.

### 5.1 Initial orientation

After localizing the user's starting checkpoint, compare the live camera view with reference images taken from known directions.

When a direction is sufficiently identifiable, establish the initial orientation.

Otherwise, ask the user to slowly rotate the phone until a recognized reference view is found.

### 5.2 Relative rotation

Use Android browser orientation and motion sensor measurements to estimate rotation relative to the established starting orientation.

For example:

- Starting orientation: facing the corridor.
- Required action: turn right.
- Expected change: approximately 90° clockwise.
- Measured change: approximately 90° clockwise.

The gyroscope is useful for tracking short rotations, while absolute compass heading can serve as optional supporting information.

The system must account for the fact that the phone can rotate independently of the user's body.

### 5.3 Visual confirmation after turning

After detecting a rotation, compare the current camera frame with reference images recorded in the expected outgoing direction.

Only when orientation evidence agrees should the system announce that the phone is facing the intended path.

### 5.4 Orientation uncertainty

Orientation uncertainty is independent of location uncertainty.

For example, the system might recognize the Stage but not know whether the user is facing the Entrance or the Drinks Area.

In that situation, the app must not issue an unsupported left/right instruction.

Instead, it should request another camera view or orientation confirmation.

---

## 6. Localization Uncertainty Management

**Goal:** Prevent the navigation controller from using unreliable visual or orientation estimates.

We will use three confidence states.

### Confirmed

Visual observations consistently identify a checkpoint, the match is sufficiently distinctive, and the result is compatible with the navigation history.

The controller may update the confirmed location.

Directional guidance additionally requires valid orientation information.

### Uncertain

Possible causes:

- Multiple visually similar locations.
- Insufficient image observations.
- Inconsistent candidate matches.
- Uncertain checkpoint proximity.
- Uncertain phone orientation.

The controller keeps the last confirmed location and refrains from advancing the route.

The app may request additional observations or a manually confirmed checkpoint.

### Lost

Possible causes:

- No recognizable reference landmarks.
- Contradictory observations.
- Camera permission or sensor failure.
- Stale localization results.
- Significant deviation from the expected route.

The system stops directional guidance and asks the user to stop moving, reorient the camera, or request assistance.

### Confidence estimation

Use a combination of:

1. Absolute image similarity.
2. Separation between the best matching candidates.
3. Temporal agreement.
4. Graph consistency.
5. Orientation agreement.

Similarity scores are not calibrated probabilities.

Thresholds should be selected empirically using images recorded at the venue, including difficult and ambiguous cases.

We should prioritize preventing false confirmations over maximizing the number of detections.

---

## 7. Route Planning

**Goal:** Compute a path from the confirmed starting checkpoint to the selected destination.

Use BFS on the verified semantic graph.

For the MVP:

- All connections must be verified and traversable.
- Only explicitly available directed edges may be used.
- The route consists of an ordered sequence of checkpoints.
- Each transition has associated guidance information.
- If no verified route exists, the app reports that navigation is unavailable.

We do not need a VLM for route planning.

For example, an illustrative route might be:

Entrance → Stage → Drinks Area

The actual path must come from the venue graph.

Future versions can introduce weighted graph search to account for distance, accessibility requirements, and route complexity.

---

## 8. Navigation Controller

**Goal:** Coordinate localization, orientation, route progression and feedback.

Use a deterministic state machine.

### Main states

**IDLE**  
No navigation session is active.

**LOCALIZING**  
The application attempts to identify the starting checkpoint.

**ALIGNING**  
The starting checkpoint is identified, but the required orientation has not yet been confirmed.

**GUIDING**  
The system has an active route and provides instructions toward the next checkpoint.

**CHECKPOINT CONFIRMATION**  
The expected checkpoint has been recognized. The controller validates that the user has reached it before advancing.

**ARRIVED**  
The final destination checkpoint has been confirmed.

**PAUSED / LOST**  
Navigation is paused because of uncertainty, system failure or a user request.

### Controller responsibilities

- Maintain the current and next expected checkpoint.
- Maintain the chosen destination.
- Track location and orientation uncertainty separately.
- Verify that observations are plausible.
- Prevent accidental checkpoint skipping.
- Advance through the route after confirmation.
- Trigger guidance only when the required conditions are satisfied.
- Prevent repetitive or contradictory instructions.
- Stop or pause navigation when necessary.

If the user appears to be off-route, the controller should first establish a reliable location. Replanning may occur only over verified graph connections.

---

## 9. Voice and Haptic Guidance

**Goal:** Provide accessible, concise, and understandable navigation instructions directly from Android Chrome.

### 9.1 Voice feedback

Use browser text-to-speech.

Examples:

- "You are near the entrance."
- "The drinks station is on your right."
- "Turn left to face the corridor."
- "The next checkpoint is the stage."
- "Your location is uncertain. Please stop."
- "You have reached your destination."

Instructions should be short, event-driven, and based on verified information.

Do not continuously generate speech for every incoming camera frame.

Include options to repeat the last instruction, pause navigation, and stop navigation.

### 9.2 Haptic feedback

Use Android Chrome's vibration capabilities.

Design distinct patterns for:

- Left turn.
- Right turn.
- Checkpoint confirmed.
- Destination reached.
- Stop or localization uncertainty.

Test the patterns on the actual Android phone.

Avoid excessive vibration and ensure the patterns remain distinguishable.

Voice remains available when vibration is unsupported or disabled.

### 9.3 Feedback policy

The navigation controller produces structured guidance events.

A separate feedback component translates each event into speech and vibration.

This keeps the navigation logic independent from the output interface and allows feedback patterns to be adjusted without changing routing.

---

## 10. Webapp Integration

The navigation stack will be integrated into the React + TypeScript application.

The phone will access the deployed frontend through HTTPS.

### Frontend responsibilities

- Start and manage the rear-facing camera.
- Sample frames for visual localization.
- Read available orientation sensors.
- Allow destination selection.
- Display the current navigation state.
- Deliver voice and haptic feedback.
- Provide large, accessible navigation controls.
- Pause and resume navigation.
- Keep the screen active when the browser supports it.

### Backend responsibilities

- Store the semantic map and reference-image embeddings.
- Process live camera frames through the selected visual encoder.
- Perform similarity matching.
- Return ranked landmark candidates and scores.

For the MVP, graph routing and the navigation state machine can run entirely in TypeScript within the browser.

No dedicated navigation backend is required.

### Integration inputs

The navigation module expects:

1. A verified semantic-spatial graph.
2. Reference landmarks and directional images.
3. Live visual localization observations.
4. Phone orientation measurements.
5. A destination selected by the user.

### Integration outputs

The navigation module produces:

1. Current confirmed checkpoint.
2. Current navigation state.
3. Next expected checkpoint.
4. Route progress.
5. Location and orientation uncertainty.
6. Voice and haptic guidance events.

---

## 11. Development and Testing Strategy

Implementation should proceed independently of the mapping team where possible.

### Milestone 1 — Navigation interfaces

Define the contracts for the map, localization observations, orientation estimates and guidance events.

**Deliverable:** agreed module interfaces.

### Milestone 2 — Routing and state machine

Implement verified-edge routing and navigation progression using a small example graph.

Use simulated landmark observations for testing.

**Deliverable:** working navigation logic independent of computer vision.

### Milestone 3 — Android feedback

Integrate speech synthesis, vibration and navigation controls.

Test in Android Chrome through an HTTPS deployment.

**Deliverable:** voice and haptic feedback triggered by navigation events.

### Milestone 4 — Initial map integration

Once the teammate's video is available:

- Extract and label reference checkpoints.
- Generate the semantic-spatial graph.
- Verify the topology and instructions.
- Associate directional reference images.
- Generate landmark embeddings.

**Deliverable:** first navigable venue map.

### Milestone 5 — Real visual localization

Connect the phone camera with the embedding-based localization service.

Test visual matching at the actual checkpoint locations.

**Deliverable:** stable landmark recognition integrated into the navigation controller.

### Milestone 6 — Orientation-aware navigation

Integrate relative phone rotation measurements and directional image matching.

Test turning instructions at a verified intersection.

**Deliverable:** navigation with one visually confirmed turn.

### Milestone 7 — End-to-end validation

Test a complete physical route through the venue.

Verify:

- Correct starting location.
- Correct route.
- Correct directional instructions.
- Correct waypoint progression.
- Arrival recognition.
- Appropriate handling of uncertain localization.
- Working voice and haptics.
- Acceptable response latency.

**Deliverable:** complete demo-ready navigation flow.

---

## 12. Hackathon Demo — 60 Seconds

The map and reference embeddings should already be prepared before the demonstration.

**0–10 seconds — Initialize**

Open the webapp on Android Chrome, activate the camera and recognize the starting checkpoint.

**10–20 seconds — Select destination**

Choose the Drinks Area or another nearby destination.

The application calculates the route.

**20–35 seconds — Initial guidance**

The system announces the next checkpoint and any verified directional instruction.

**35–50 seconds — Navigation and turn**

Demonstrate a short route with one turn.

The app detects phone rotation, verifies the outgoing camera view and confirms the next checkpoint.

**50–60 seconds — Arrival**

The final checkpoint is recognized and the app announces arrival with voice and vibration.

The physical route must be short enough to fit the demonstration time. The prototype will be demonstrated by a sighted tester under supervision.

---

## 13. Priorities

### Essential for the MVP

- Verified semantic navigation graph.
- Pretrained visual embedding model.
- Landmark similarity matching.
- Graph-based routing.
- Temporal observation validation.
- Navigation state machine.
- Location uncertainty handling.
- Voice and haptic feedback.
- End-to-end operation through Android Chrome.

### High priority

- Orientation-aware reference images.
- Relative phone rotation detection.
- Visual confirmation after turning.
- Distinct location and orientation uncertainty.
- Checkpoint proximity confirmation.

### Future extensions

- Continuous visual-inertial localization.
- Full SLAM.
- Metric positioning and distance-to-waypoint estimation.
- Dynamic obstacle avoidance.
- Automatic recovery from unexpected route deviations.
- Multi-floor navigation.
- More sophisticated accessible route optimization.
- Automatic map merging across repeated walkthroughs.

These should not block the hackathon MVP.

---

## 14. Risks and Fallback Strategies

| Risk | Mitigation |
|---|---|
| Incorrect VLM-inferred connections | Manual verification of the first map |
| Similar-looking landmarks | Multiple reference images and temporal consistency |
| Landmark visible from far away | Close-range checkpoint references and explicit confirmation |
| Orientation sensor drift | Visual verification after turns |
| Wrong directional instruction | Require verified approach orientation |
| Slow model inference | Precompute embeddings, process selected frames, avoid repeated VLM calls |
| Localization becomes uncertain | Pause directional guidance and request relocalization |
| Backend or network unavailable | Keep a separately labeled simulated-navigation test mode |
| Vibration unavailable | Continue with spoken feedback |

The prototype will not provide validated continuous localization, drop-off detection, or dynamic obstacle avoidance. It must not be presented as a safe replacement for established mobility aids.

---

## 15. Definition of Done

The navigation stack will be considered ready for the hackathon when:

1. The verified map can be loaded into the React webapp.
2. The user can select a destination.
3. The phone camera can recognize the mapped starting checkpoint.
4. The application calculates a valid route using verified connections.
5. The controller issues the appropriate navigation instructions.
6. At least one directional turn can be verified using visual and/or orientation evidence.
7. The application recognizes the final checkpoint.
8. Voice and haptic feedback work on the demonstration Android phone.
9. Uncertain observations do not trigger unsupported directional instructions.
10. A sighted tester can complete the supervised 60-second demonstration using real camera observations.

**Final architecture:**

Video-derived semantic and spatial map + Visual Place Recognition + Phone orientation + Graph routing + Navigation state machine + Voice/haptic feedback.

No VLA is required for the initial version, and a VLM is primarily used during mapping rather than during continuous navigation.
