<!-- Adapted from tungsten-united/demo-repository@c5dc26f (branch navigation), by LPA. Aligned with Orient on 2026-10-10: the phone app talks only to Orient's orchestrator, which calls nav-api; there is no navigation state machine. The pipeline it describes is nav/ (see README). -->

# Project Overview — AI-Powered Indoor Navigation for Visually Impaired People

## 1. Main Idea

We are developing an **AI-powered indoor navigation web application designed to help blind and visually impaired people navigate unfamiliar indoor environments**.

The application allows a building or indoor space to be mapped through a simple smartphone video recording. AI processes the video to create a lightweight semantic and spatial map, identifying important landmarks, destinations, corridors, intersections, and their connections.

Once the environment has been mapped, users open the application in their smartphone's browser, say where they want to go, and receive navigation guidance through **voice instructions** (haptic feedback is a later option).

Our vision is to make indoor navigation more accessible without requiring specialized hardware, LiDAR sensors, or the installation of a native mobile application.

**Core concept: Scan a place once. Make it navigable for everyone.**

## 2. Problem We Want to Solve

Outdoor navigation is widely supported by GPS-based applications such as Google Maps. However, indoor navigation remains challenging, particularly for people with visual impairments.

GPS is often unreliable inside buildings, while indoor spaces contain corridors, doors, intersections, and obstacles that conventional navigation applications cannot easily interpret.

Many indoor navigation solutions also require specialized infrastructure, manually constructed maps, or dedicated hardware.

We want to investigate a simpler, more scalable approach using smartphone cameras, AI-based visual perception, and reusable semantic maps.

## 3. How the Application Works

The application has two main operational modes.

### Mode A — Mapping an Environment

A person records a video while walking through an indoor environment.

For example, during the hackathon, we will record the venue, which contains:

- Main entrance
- Stage
- Kitchen
- Drinks area
- Corridors, intersections, and other relevant navigation checkpoints

The video is processed with AI to extract semantic and spatial information: vision models select and match frames, and a Vision-Language Model (Claude) writes the graph from them, the walker's voice notes and the motion data.

The system identifies important locations, generates reference images, proposes traversable connections, and extracts relative spatial relationships.

For example, from a particular recorded viewpoint:

- The drinks station is on the right.
- A corridor continues straight ahead.
- The entrance is behind the camera.

This information is organized into an **orientation-aware semantic navigation graph**.

The graph contains navigation nodes, verified connections, reference images, and directional relationships. Automatically inferred information is meant to be reviewed before being used for navigation. For the supervised demo, Orient's orchestrator also uses connections that were walked but not yet verified.

The goal is not to reconstruct a detailed 3D environment, but to produce a lightweight, reusable representation suitable for navigation.

### Mode B — Navigating the Environment

Once a map is available, a visually impaired user opens the application using an HTTPS link on a smartphone.

The user says a destination, such as "Kitchen" or "Drinks Area".

The phone sends camera frames to Orient's orchestrator, a server, which asks nav-api to compare them with previously recorded reference images (**Visual Place Recognition, VPR**). The phone never calls nav-api itself.

A pretrained place-recognition encoder (MegaLoc) generates image embeddings, which are compared with those stored during mapping.

This helps estimate the user's current location within the semantic graph.

nav-api's graph routing then gives a route toward the destination over trusted connections, and the orchestrator guides the user along it.

During navigation, the system combines:

1. **Visual localization:** identifying the current checkpoint or landmark (nav-api).
2. **Orientation estimation:** determining the direction the phone is facing using visual references and available smartphone motion sensors. Not built yet.
3. **Spatial reasoning:** interpreting the directional relationships stored during mapping. Stored in the map, not used live yet.
4. **Route progression:** deciding which checkpoint the user should reach next (the orchestrator).
5. **Accessible guidance:** providing spoken instructions (the orchestrator, played by the phone); distinctive vibration patterns are a later option.

For example, the application could say:

- "You are near the entrance."
- "The drinks station is on your right."
- "Turn left to face the corridor."
- "Your next checkpoint is the stage."
- "You have reached the kitchen."

Guidance is updated as the orchestrator confirms new checkpoints.

If localization or orientation is uncertain, the application must communicate that uncertainty and avoid issuing unsupported directional instructions.

## 4. Key Technical Decisions

The user-facing application is a **mobile-first webapp** (React, TypeScript and Tailwind CSS, in tungsten-united/project-description).

It runs directly in the phone's browser through a public HTTPS URL, and talks only to Orient's orchestrator (tungsten-united/orient-orchestrator), a server that holds the session and calls nav-engine's nav-api.

We are not developing a native Android application.

The architecture uses:

- **VLM:** semantic and spatial information extraction from the initial mapping video.
- **Visual embedding model:** place recognition during live navigation (MegaLoc, in nav-api).
- **Semantic-spatial graph:** representation of locations, connections, and directional relationships.
- **Graph search:** deterministic route calculation (nav-api).
- **Smartphone sensors:** relative orientation and turn estimation (not built yet).
- **Route following in the orchestrator:** the user's last confirmed checkpoint, route progression and uncertainty handling, kept in the orchestrator's session. There is no separate navigation state machine.
- **Speech:** spoken guidance, synthesized by the orchestrator (ElevenLabs) and played by the phone, with the browser's speech as fallback.
- **Vibration API:** haptic feedback, a later option.

We do not plan to use a Vision-Language-Action (VLA) model for the initial version. Deterministic route following is more appropriate for this prototype.

## 5. Scalability

A central objective is to avoid requiring each user to map the same building independently.

An environment should be mapped once and subsequently reused by multiple users.

The architecture should therefore support:

- Multiple buildings and floors.
- Reusable semantic navigation graphs.
- Efficient visual landmark retrieval.
- Updates and corrections to existing maps.
- Additional navigation checkpoints without rebuilding the entire system.
- Future improvements to localization and accessibility-aware navigation.

The hackathon prototype will demonstrate this architecture on one venue and one floor.

## 6. Hackathon MVP

Our immediate objective is to demonstrate the complete pipeline within a **60-second live demonstration**.

The venue will be mapped before the demonstration.

During the demo, the application should:

1. Open through a normal URL on an Android smartphone.
2. Activate the camera and recognize the starting checkpoint.
3. Allow the user to say a destination.
4. Calculate a route through the verified semantic graph.
5. Provide an initial voice instruction.
6. Recognize navigation checkpoints using live camera observations.
7. Detect and verify a directional turn.
8. Update the guidance according to the user's confirmed progress.
9. Recognize the destination and announce arrival.

The demonstration should include at least one meaningful turn to showcase the integration of visual recognition, spatial information, orientation estimation, and multimodal feedback.

## 7. Scope and Safety Limitations

The first version is a proof of concept, not a validated independent mobility aid.

It will not provide reliable continuous positioning, complete obstacle detection, stair or drop-off detection, or guarantees that a path remains clear of moving obstacles.

Recognizing a landmark does not necessarily mean the user has reached it. Checkpoint progression must account for this uncertainty.

Real-world testing during the hackathon will use a sighted tester on a supervised, previously checked route.

The long-term goal is to develop a navigation assistant for visually impaired users, but independent navigation would require substantially stronger localization, perception, accessibility evaluation, and safety validation.

## 8. Current Team Responsibilities

The project is being developed by multiple team members in parallel.

**Mapping pipeline:** Another team member is responsible for recording the walkthrough video and contributing to the initial video-processing pipeline.

**Navigation stack (current development responsibility):** Our focus is designing and implementing the navigation functionality, including:

- Integration with the generated semantic-spatial graph.
- Visual Place Recognition.
- Localization uncertainty estimation.
- Phone orientation estimation and turn verification.
- Graph-based routing.
- Route following, in Orient's orchestrator.
- Guidance wording (speech output belongs to the orchestrator and the phone).
- The navigation API (nav-api) that Orient's orchestrator calls.

The navigation module should remain independent of the specific mapping and visual inference implementations, communicating through clearly defined interfaces.

## 9. Development Objective

The priority is to build a **small, functional, modular and demonstrable system**, not an unnecessarily complex autonomous navigation framework.

The first version should work with a verified map containing only a few destinations and checkpoints.

We should prioritize reliable behavior, clear interfaces, uncertainty handling, and successful end-to-end integration over sophisticated but unvalidated features.

The system should initially be developed using simulated map and localization events, then integrated with the real mapping pipeline once the recorded video and reference images are available.

**Current development stage (2026-10-10):** mapping, review and nav-api (routes, localization) are built, and Orient's orchestrator calls nav-api. Orientation estimation and turn verification are not built.
