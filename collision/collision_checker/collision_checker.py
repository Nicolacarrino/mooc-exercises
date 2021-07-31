import itertools
import random
from typing import List

from aido_schemas import Context, FriendlyPose
from dt_protocols import (
    Circle,
    CollisionCheckQuery,
    CollisionCheckResult,
    MapDefinition,
    PlacedPrimitive,
    Rectangle,
)

import shapely.geometry

__all__ = ["CollisionChecker"]

PRINT = False


class CollisionChecker:
    params: MapDefinition

    def init(self, context: Context):
        context.info("init()")

    def on_received_set_params(self, context: Context, data: MapDefinition):
        context.info("initialized")
        self.params = data

    def on_received_query(self, context: Context, data: CollisionCheckQuery):
        collided = check_collision(
            Wcoll=self.params.environment, robot_body=self.params.body, robot_pose=data.pose
        )
        result = CollisionCheckResult(collided)
        context.write("response", result)


def check_collision(
    Wcoll: List[PlacedPrimitive], robot_body: List[PlacedPrimitive], robot_pose: FriendlyPose
) -> bool:
    # This is just some code to get you started, but you don't have to follow it exactly

    # start by rototranslating the robot parts by the robot pose
    rototranslated_robot: List[PlacedPrimitive] = [PlacedPrimitive(robot_pose, x.primitive) for x in robot_body ]

    collided = check_collision_list(rototranslated_robot, Wcoll)

    # return a random choice
    return collided


def check_collision_list(A: List[PlacedPrimitive], B: List[PlacedPrimitive]) -> bool:
    # This is just some code to get you started, but you don't have to follow it exactly

    for a, b in itertools.product(A, B):
        if check_collision_shape(a, b):
            return True

    return False


def check_collision_shape(a: PlacedPrimitive, b: PlacedPrimitive) -> bool:
    # This is just some code to get you started, but you don't have to follow it exactly
    if PRINT:
        print(a)
        print(a.primitive)
    a_shap = shapely.geometry.Point(0, 0).buffer(1)
    b_shap = shapely.geometry.Point(0, 0).buffer(1)


    if isinstance(a.primitive, Circle):
        a_shap = shapely.geometry.Point(a.pose.x, a.pose.y).buffer(a.primitive.radius)
        if PRINT:
            print("a Circle")

    if isinstance(b.primitive, Circle):
        b_shap = shapely.geometry.Point(b.pose.x, b.pose.y).buffer(b.primitive.radius)
        if PRINT:
            print("b Circle")

    if isinstance(a.primitive, Rectangle):
        a_shap = get_shapely_rect(a)
        if PRINT:
            print("a Rectangle")

    if isinstance(b.primitive, Rectangle):
        b_shap = get_shapely_rect(b)
        if PRINT:
            print("b Rectangle")

    
    collision = a_shap.intersects(b_shap)
    if PRINT:
        print("collision: ",collision)
        print()
        print()

    # if isinstance(a, Circle) and isinstance(b, Circle):
    #     ...
    # if isinstance(a, Circle) and isinstance(b, Rectangle):
    # for now let's return a random guess

    return collision

def get_shapely_rect(pprim: PlacedPrimitive):
    rect = pprim.primitive

    origin = (pprim.pose.x, pprim.pose.y)

    shapely_rect = shapely.geometry.box(rect.xmin, rect.ymin, rect.xmax, rect.ymax)
    shapely_rect = shapely.affinity.translate(shapely_rect, xoff=pprim.pose.x, yoff=pprim.pose.y, zoff=0.0)
    shapely_rect = shapely.affinity.rotate(shapely_rect, pprim.pose.theta_deg, origin=origin)

    return shapely_rect