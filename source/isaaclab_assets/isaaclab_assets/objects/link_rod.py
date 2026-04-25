from __future__ import annotations
import os
import tempfile
from dataclasses import field

import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

import os

@configclass
class LinkedRodObjectCfg(ArticulationCfg):

    joint_positions: list = field(default_factory=lambda: [
        (0.0, 0.0, 0.0),
        (0.02, 0.06, 0.0),
        (0.04, 0.09, 0.0),
        (0.06, 0.12, 0.0),
    ])
    link_radius: float = 0.004
    joint_type: str = "fixed"

    prim_path: str = "{ENV_REGEX_NS}/Object"
    spawn: sim_utils.UsdFileCfg = None
    init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.8),
        rot=(1.0, 0.0, 0.0, 0.0),
    )
    actuators: dict = None

    def __post_init__(self):
        pts = [np.array(p, dtype=float) for p in self.joint_positions]
        assert len(pts) >= 2, "joint_positions は最低2点必要です"

        n_links = len(pts) - 1

        # ジョイント名リストを init_state に反映
        joint_pos_dict = {f"rod_joint_{i+1}": 0.0 for i in range(n_links - 1)}
        joint_vel_dict = {f"rod_joint_{i+1}": 0.0 for i in range(n_links - 1)}
        self.init_state = ArticulationCfg.InitialStateCfg(
            pos=self.init_state.pos,
            rot=self.init_state.rot,
            joint_pos=joint_pos_dict,
            joint_vel=joint_vel_dict,
        )

        usd_path = self._generate_usd(pts)

        self.spawn = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        )

        stiffness = 0.0 if self.joint_type == "revolute" else 1e6
        damping   = 0.5 if self.joint_type == "revolute" else 1e4
        if n_links > 1:
            self.actuators = {
                "rod_joints": ImplicitActuatorCfg(
                    joint_names_expr=["rod_joint_.*"],
                    stiffness=stiffness,
                    damping=damping,
                )
            }
        else:
            self.actuators = {}

    # ------------------------------------------------------------------
    def _generate_usd(self, pts: list[np.ndarray]) -> str:
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

        n_links = len(pts) - 1
        usd_path = os.path.join(
            tempfile.gettempdir(),
            f"linked_rod_{n_links}links_{self.joint_type}.usda",
        )

        stage = Usd.Stage.CreateNew(usd_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        root_path = "/LinkedRod"
        root_xform = UsdGeom.Xform.Define(stage, root_path)
        UsdPhysics.ArticulationRootAPI.Apply(root_xform.GetPrim())

        link_paths = []

        for i in range(n_links):
            p_start = pts[i]
            p_end   = pts[i + 1]
            vec     = p_end - p_start
            length  = float(np.linalg.norm(vec))
            center  = (p_start + p_end) / 2.0

            link_name = f"rod_link_{i}"
            link_path = f"{root_path}/{link_name}"

            xform = UsdGeom.Xform.Define(stage, link_path)

            # ① 平行移動
            xform.AddTranslateOp().Set(Gf.Vec3d(*center.tolist()))

            # ② 回転（Y軸 → vec方向）をクォータニオンで設定
            quat = _rotation_from_y_to_vec(vec)
            xform.AddOrientOp().Set(quat)

            # 円柱（Y軸方向がUSDデフォルト）
            shape_path = f"{link_path}/CylinderShape"
            cyl = UsdGeom.Cylinder.Define(stage, shape_path)
            cyl.GetRadiusAttr().Set(self.link_radius)
            cyl.GetHeightAttr().Set(length)
            cyl.GetAxisAttr().Set("Y")
            UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

            UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
            mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
            mass_api.GetMassAttr().Set(0.05 / n_links)

            link_paths.append(link_path)

        # --- ジョイント生成 ---
        for i in range(n_links - 1):
            joint_name = f"rod_joint_{i + 1}"
            joint_path = f"{link_paths[i + 1]}/{joint_name}"

            p_joint   = pts[i + 1]

            if self.joint_type == "fixed":
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            else:
                joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                joint.GetAxisAttr().Set("X")
                joint.GetLowerLimitAttr().Set(-90.0)
                joint.GetUpperLimitAttr().Set(90.0)

            joint.GetBody0Rel().SetTargets([Sdf.Path(link_paths[i])])
            joint.GetBody1Rel().SetTargets([Sdf.Path(link_paths[i + 1])])

            center_i  = (pts[i]     + pts[i + 1]) / 2.0
            center_i1 = (pts[i + 1] + pts[i + 2]) / 2.0

            # ---- 各リンクの回転行列を取得して逆変換 ----
            def local_pos(p_world, center, vec):
                """ワールド座標をリンクローカル座標に変換"""
                v = vec / np.linalg.norm(vec)
                y = np.array([0.0, 1.0, 0.0])
                axis = np.cross(y, v)
                sin_a = np.linalg.norm(axis)
                cos_a = np.dot(y, v)
                if sin_a < 1e-6:
                    R = np.eye(3) if cos_a > 0 else np.diag([1, -1, -1])
                else:
                    axis = axis / sin_a
                    angle = np.arctan2(sin_a, cos_a)
                    # ロドリゲスの回転行列
                    K = np.array([
                        [0,        -axis[2],  axis[1]],
                        [axis[2],   0,       -axis[0]],
                        [-axis[1],  axis[0],  0      ],
                    ])
                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                # ローカル座標 = R^T * (world - center)
                return R.T @ (p_world - center)

            vec_i  = pts[i + 1] - pts[i]
            vec_i1 = pts[i + 2] - pts[i + 1]

            lp0 = local_pos(p_joint, center_i,  vec_i)
            lp1 = local_pos(p_joint, center_i1, vec_i1)

            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*lp0.tolist()))
            joint.GetLocalPos1Attr().Set(Gf.Vec3f(*lp1.tolist()))

        stage.SetDefaultPrim(root_xform.GetPrim())
        stage.Save()
        return usd_path
    
def _rotation_from_y_to_vec(vec: np.ndarray):
    """
    USD Cylinder(Y軸方向)をvec方向に向けるクォータニオンを返す。
    Gf.Rotation は使わず numpy で直接 quaternion を計算する。
    戻り値: Gf.Quatf
    """
    from pxr import Gf

    y = np.array([0.0, 1.0, 0.0])
    v = vec / np.linalg.norm(vec)

    axis = np.cross(y, v)
    sin_a = float(np.linalg.norm(axis))
    cos_a = float(np.dot(y, v))

    if sin_a < 1e-6:
        if cos_a > 0:
            # 修正前: Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0))
            return Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        else:
            # 修正前: Gf.Quatd(0.0, Gf.Vec3d(1.0, 0.0, 0.0))
            return Gf.Quatf(0.0, Gf.Vec3f(1.0, 0.0, 0.0))

    # 正規化した回転軸
    axis = axis / sin_a

    # 半角公式でクォータニオンを構築
    half_angle = float(np.arctan2(sin_a, cos_a)) / 2.0
    w = float(np.cos(half_angle))
    s = float(np.sin(half_angle))

    return Gf.Quatf(w, Gf.Vec3f(
        float(axis[0]) * s,
        float(axis[1]) * s,
        float(axis[2]) * s,
    ))
