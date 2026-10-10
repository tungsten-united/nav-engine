<!-- From tungsten-united/demo-repository@c5dc26f (branch navigation), by LPA. Kept verbatim; the pipeline it describes is now nav/ (see README). -->

# Project Overview — AI-Powered Indoor Navigation for Visually Impaired People

## 1. Main Idea

We are developing an **AI-powered indoor navigation web application designed to help blind and visually impaired people navigate unfamiliar indoor environments**.

The application allows a building or indoor space to be mapped through a simple smartphone video recording. AI processes the video to create a lightweight semantic and spatial map, identifying important landmarks, destinations, corridors, intersections, and their connections.

Once the environment has been mapped, users can access the application through a browser on their smartphones, select a destination, and receive navigation guidance through **voice instructions and haptic feedback**.

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

The video is processed using AI, particularly a Vision-Language Model (VLM), to extract semantic and spatial information.

The system identifies important locations, generates reference images, proposes traversable connections, and extracts relative spatial relationships.

For example, from a particular recorded viewpoint:

- The drinks station is on the right.
- A corridor continues straight ahead.
- The entrance is behind the camera.

This information is organized into an **orientation-aware semantic navigation graph**.

The graph contains navigation nodes, verified connections, reference images, and directional relationships. Automatically inferred information is reviewed before being used for navigation.

The goal is not to reconstruct a detailed 3D environment, but to produce a lightweight, reusable representation suitable for navigation.

### Mode B — Navigating the Environment

Once a map is available, a visually impaired user opens the application using an HTTPS link on an Android smartphone.

The user selects a destination, such as "Kitchen" or "Drinks Area".

The application activates the smartphone camera and uses **Visual Place Recognition (VPR)** to compare live observations with previously recorded reference images.

A pretrained visual encoder generates image embeddings, which are compared with those stored during mapping.

This helps estimate the user's current location within the semantic graph.

A graph-based routing algorithm then identifies a route toward the selected destination using verified traversable connections.

During navigation, the system combines:

1. **Visual localization:** identifying the current checkpoint or landmark.
2. **Orientation estimation:** determining the direction the phone is facing using visual references and available smartphone motion sensors.
3. **Spatial reasoning:** interpreting the directional relationships stored during mapping.
4. **Route progression:** deciding which verified checkpoint the user should reach next.
5. **Accessible guidance:** providing spoken instructions and distinctive vibration patterns.

For example, the application could say:

- "You are near the entrance."
- "The drinks station is on your right."
- "Turn left to face the corridor."
- "Your next checkpoint is the stage."
- "You have reached the kitchen."

Guidance is updated as the application confirms new checkpoints and orientation changes.

If localization or orientation is uncertain, the application must communicate that uncertainty and avoid issuing unsupported directional instructions.

## 4. Key Technical Decisions

The application is a **mobile-first webapp**, developed using React, TypeScript, and Tailwind CSS.

It must run directly in Android Chrome through a public HTTPS URL.

We are not developing a native Android application.

The proposed architecture uses:

- **VLM:** semantic and spatial information extraction from the initial mapping video.
- **Visual embedding model:** image similarity and place recognition during live navigation.
- **Semantic-spatial graph:** representation of locations, connections, and directional relationships.
- **Graph search:** deterministic route calculation.
- **Smartphone sensors:** relative orientation and turn estimation.
- **Navigation state machine:** route progression, checkpoint confirmation, and uncertainty management.
- **Web Speech API:** spoken guidance.
- **Vibration API:** haptic feedback on supported Android devices.

We do not plan to use a Vision-Language-Action (VLA) model for the initial version. A deterministic navigation controller is more appropriate for this prototype.

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
3. Allow the user to choose a destination.
4. Calculate a route through the verified semantic graph.
5. Provide an initial voice instruction and appropriate vibration.
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
- Navigation state management.
- Voice and haptic feedback.
- Integration into the React webapp.

The navigation module should remain independent of the specific mapping and visual inference implementations, communicating through clearly defined interfaces.

## 9. Development Objective

The priority is to build a **small, functional, modular and demonstrable system**, not an unnecessarily complex autonomous navigation framework.

The first version should work with a verified map containing only a few destinations and checkpoints.

We should prioritize reliable behavior, clear interfaces, uncertainty handling, and successful end-to-end integration over sophisticated but unvalidated features.

The system should initially be developed using simulated map and localization events, then integrated with the real mapping pipeline once the recorded video and reference images are available.

**Current development stage:** Architecture and implementation planning. Do not generate or execute implementation code until explicitly requested.
