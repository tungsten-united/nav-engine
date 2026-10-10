"""Navigation side: routing on a place's map and following a user along a route (docs/navigation_stack.md).

Reference implementations in Python, small and causal so they port to the TypeScript app: `route.plan_route`
(verified edges only, by default) and `pdr.GraphTracker` (steps + heading -> progress along the route).
"""
