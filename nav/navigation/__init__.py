"""Navigation side: routing on a place's map and following a user along a route (docs/navigation_stack.md).

Small and causal. nav-api serves `route.plan_route` (verified edges only, by default) and `localize.localize` to
Orient's orchestrator; `pdr.GraphTracker` (steps + heading -> progress along the route) is reference code that
nothing live runs.
"""
