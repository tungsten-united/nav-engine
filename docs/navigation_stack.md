<!-- Adapted from tungsten-united/demo-repository@c5dc26f (branch navigation), by LPA. Aligned with Orient on 2026-10-10: the phone app talks only to Orient's orchestrator, which calls nav-api; there is no navigation state machine. The pipeline it describes is nav/ (see README). -->

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

The phone webapp sends camera frames to Orient's orchestrator, which asks nav-api which node of the generated map they show.

The orchestrator uses nav-api's routes over the graph to determine which waypoint to reach next and what guidance to provide. Phone orientation measurements are not used yet.

The orchestrator turns guidance into speech, which the phone plays. Haptic feedback is a later option.

**Main pipeline:**

Mapping video → Semantic-spatial map → Live camera (phone) → Visual localization (nav-api) → Orientation estimation (not built) → Route following (orchestrator, routes from nav-api) → Voice (orchestrator and phone).

---

## 2. Interface Between Mapping and Navigation

Before receiving the video, we need to agree with the mapping team on the structure of the generated map. The agreed format is `nav/schemas/map.py` (JSON Schema: `docs/map.schema.json`).

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

Chosen: **MegaLoc**, a specialized VPR model. On the venue, DINOv2's general features said "same kind of scene" rather than "same place", so DINOv2 is only used during mapping, to tell new views from near-duplicates.

Its recognition performance on held-out venue images is still to be evaluated.

During mapping, generate and save embeddings for reference images.

During navigation, nav-api's `localize` embeds the 1–4 camera frames Orient's orchestrator sends and compares them against stored references.

The output should include:

- Most likely checkpoint candidates.
- Similarity scores.
- Associated viewing direction, where identifiable.
- Observation timestamp (the caller's: nav-api is stateless).

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

nav-api's `localize` covers the strongest match, the margin, the previously confirmed location (given as `previous`) and graph plausibility (one edge away). Agreement across consecutive observations is the orchestrator's, and is not built yet.

### 4.3 Checkpoint arrival

Recognizing a landmark from a distance does not mean the user has reached its checkpoint.

For the MVP, use reference images captured from an established checkpoint position. Require additional evidence of proximity or explicit checkpoint confirmation before advancing. As of 2026-10-10, the orchestrator advances on a single `confirmed` result.

We will not attempt centimeter-level localization or continuous metric position estimation.

---

## 5. User Orientation Estimation

**Goal:** Determine the direction the phone is facing relative to the recorded navigation graph.

We will combine relative phone rotation with visual orientation verification.

Not built yet. nav-api's `localize` already reports each matched reference's camera heading and, given the phone's compass heading (`heading_deg`), the difference; the orchestrator does not send the heading yet.

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

In that situation, the orchestrator must not issue an unsupported left/right instruction.

Instead, it should request another camera view or orientation confirmation.

---

## 6. Localization Uncertainty Management

**Goal:** Prevent the orchestrator's route following from using unreliable visual or orientation estimates.

nav-api's `localize` returns three confidence states.

### Confirmed

Visual observations consistently identify a checkpoint, the match is sufficiently distinctive, and the result is compatible with the navigation history.

The orchestrator may update the confirmed location.

Directional guidance additionally requires valid orientation information.

### Uncertain

Possible causes:

- Multiple visually similar locations.
- Insufficient image observations.
- Inconsistent candidate matches.
- Uncertain checkpoint proximity.
- Uncertain phone orientation.

The orchestrator keeps the last confirmed location and refrains from advancing the route.

The orchestrator may ask the user, through the phone, for a clearer view or a manually confirmed checkpoint.

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

nav-api's `localize` uses 1, 2 and 4; its thresholds are placeholders until tuned on venue images. 3 and 5 are not built.

---

## 7. Route Planning

**Goal:** Compute a path from the confirmed starting checkpoint to the selected destination.

nav-api's `POST /route` finds the shortest route by distance over trusted edges, in the direction they may be walked.

For the MVP:

- Real users should be routed over verified connections only (nav-api's default). For the supervised demo, Orient's orchestrator also uses `observed` edges, walked but not yet verified, until the demo route is verified.
- Only explicitly available directed edges may be used.
- The route consists of an ordered sequence of checkpoints.
- Each transition has associated guidance information.
- If no trusted route exists, nav-api answers `found: false` and the orchestrator waits.

We do not need a VLM for route planning.

For example, an illustrative route might be:

Entrance → Stage → Drinks Area

The actual path must come from the venue graph.

Future versions can weigh accessibility requirements and route complexity as well as distance.

---

## 8. Route Following (Orient's orchestrator)

**Goal:** Coordinate localization, orientation, route progression and feedback.

There is no separate navigation state machine. Orient's orchestrator keeps each session's state: the destination, the user's last confirmed checkpoint and the previous guidance. For every evaluated frame it calls nav-api's `localize`, then `route` from the confirmed checkpoint, and turns the route's first hop into guidance. The phone app has its own client states (listening, speaking, stopped and so on), which are about the conversation, not the route. The contract is tungsten-united/project-description `docs/contracts.md` §2.

### Responsibilities (design; the contract says what is built)

- Maintain the current and next expected checkpoint.
- Maintain the chosen destination.
- Track location and orientation uncertainty separately.
- Verify that observations are plausible.
- Prevent accidental checkpoint skipping.
- Advance through the route after confirmation.
- Trigger guidance only when the required conditions are satisfied.
- Prevent repetitive or contradictory instructions.
- Stop or pause navigation when necessary.

If the user appears to be off-route, the orchestrator should first establish a reliable location. Replanning may occur only over trusted graph connections.

---

## 9. Voice and Haptic Guidance

**Goal:** Provide accessible, concise, and understandable navigation instructions directly from Android Chrome.

### 9.1 Voice feedback

Orient's orchestrator writes the sentence (the route hop's instruction, or a template) and streams it as speech (ElevenLabs). The phone plays it, with browser text-to-speech as fallback.

Examples:

- "You are near the entrance."
- "The drinks station is on your right."
- "Turn left to face the corridor."
- "The next checkpoint is the stage."
- "Your location is uncertain. Please stop."
- "You have reached your destination."

Instructions should be short, event-driven, and based on verified information.

Do not continuously generate speech for every incoming camera frame. The orchestrator speaks only when its output changes and the change is worth saying.

Include options to repeat the last instruction, pause navigation, and stop navigation. Orient has Start/Stop, and the orchestrator repeats an unchanged instruction after a few seconds.

### 9.2 Haptic feedback

A later option (project-description card S07): Android Chrome's vibration capabilities.

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

The orchestrator produces structured outputs (action, direction, checkpoint, uncertainty).

A separate step turns each output into a sentence and speech (and later vibration).

This keeps the navigation logic independent from the output interface and allows feedback patterns to be adjusted without changing routing.

---

## 10. Integration with Orient

Three parts, each in its own repository.

### Phone webapp (tungsten-united/project-description, `apps/web`)

- Start and manage the rear-facing camera, and send frames to the orchestrator.
- Record the spoken destination.
- Read available orientation sensors and send a motion summary.
- Play the guidance speech.
- Provide large, accessible Start/Stop controls.
- Keep the screen active when the browser supports it.

It talks only to the orchestrator.

### Orient's orchestrator (tungsten-united/orient-orchestrator)

- Hold the client and session: the destination and the user's last confirmed checkpoint.
- Call nav-api for every evaluated frame: `localize`, then `route`.
- Turn the result into guidance, decide whether to speak it, and synthesize the speech.

### nav-api (this repository)

- Serve the published semantic maps and reference-image embeddings, read-only.
- Process live camera frames through the visual encoder (on the inference host) and match them.
- Return ranked checkpoint candidates, scores and a confidence status (`localize`).
- Return routes over trusted edges, with their spoken instructions (`route`).

nav-api is stateless and server to server: the phone never calls it.

### Integration inputs (nav-api)

1. A reviewed semantic-spatial graph (the published map).
2. Reference landmarks and directional images.
3. Live camera frames, the last confirmed checkpoint, and optionally the expected checkpoint and the compass heading.
4. A start and a destination checkpoint.

### Integration outputs

nav-api returns ranked checkpoints with a confirmed / uncertain / lost status, and routes. The orchestrator derives the rest:

1. Current confirmed checkpoint.
2. Next expected checkpoint.
3. Route progress.
4. Location uncertainty (orientation uncertainty is not built).
5. Voice guidance.

---

## 11. Development and Testing Strategy

Implementation should proceed independently of the mapping team where possible.

Status (2026-10-10): milestones 1, 2 and 4 are built (the map and nav-api formats, routing in nav-api and route following in the orchestrator, the Itnig map with no verified edge yet). 3 and 5 are written in Orient's phone app and orchestrator but not yet tested at the venue. 6 and 7 are open.

### Milestone 1 — Navigation interfaces

Define the contracts for the map, localization observations, orientation estimates and guidance events.

**Deliverable:** agreed module interfaces.

### Milestone 2 — Routing and route following

Implement verified-edge routing and navigation progression using a small example graph.

Use simulated landmark observations for testing.

**Deliverable:** working navigation logic independent of computer vision.

### Milestone 3 — Android feedback

Integrate speech output, navigation controls and, later, vibration.

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

Connect the phone camera, through the orchestrator, with nav-api's localization.

Test visual matching at the actual checkpoint locations.

**Deliverable:** stable landmark recognition integrated into the orchestrator's route following.

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

Say "Drinks Area" or another nearby destination.

nav-api calculates the route.

**20–35 seconds — Initial guidance**

The system announces the next checkpoint and any verified directional instruction.

**35–50 seconds — Navigation and turn**

Demonstrate a short route with one turn.

The app detects phone rotation, verifies the outgoing camera view and confirms the next checkpoint.

**50–60 seconds — Arrival**

The final checkpoint is recognized and the app announces arrival by voice.

The physical route must be short enough to fit the demonstration time. The prototype will be demonstrated by a sighted tester under supervision.

---

## 13. Priorities

### Essential for the MVP

- Verified semantic navigation graph.
- Pretrained visual embedding model.
- Landmark similarity matching.
- Graph-based routing.
- Temporal observation validation.
- Route following in the orchestrator.
- Location uncertainty handling.
- Voice feedback.
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
- Haptic feedback.

Automatic map merging across repeated walkthroughs, once listed here, is built: more walks of a place improve the same map.

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

1. Orient's orchestrator uses the verified map through nav-api.
2. The user can say a destination.
3. The phone camera can recognize the mapped starting checkpoint.
4. nav-api calculates a valid route using verified connections.
5. The orchestrator issues the appropriate navigation instructions.
6. At least one directional turn can be verified using visual and/or orientation evidence.
7. The application recognizes the final checkpoint.
8. Voice feedback works on the demonstration phone (haptics if S07 lands).
9. Uncertain observations do not trigger unsupported directional instructions.
10. A sighted tester can complete the supervised 60-second demonstration using real camera observations.

**Final architecture:**

Video-derived semantic and spatial map + Visual Place Recognition + Phone orientation + Graph routing (nav-api) + Route following in Orient's orchestrator + Voice feedback.

No VLA is required for the initial version, and a VLM is primarily used during mapping rather than during continuous navigation.
