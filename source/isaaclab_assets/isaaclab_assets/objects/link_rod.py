from __future__ import annotations
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass


@configclass
class LinkedRodObjectCfg(ArticulationCfg):
    """
    3つのリジッドボディを固定ジョイントで繋いだ棒状オブジェクトのCfg。
    各リンクは円柱(Cylinder)形状。

    構造:
        rod_link_0 (root)
            └─ rod_joint_1 (fixed / revolute) ─ rod_link_1
                └─ rod_joint_2 (fixed / revolute) ─ rod_link_2

    Parameters:
        link_length: 各リンクの長さ [m]
        link_radius: 各リンクの半径 [m]
        joint_type:  "fixed" で剛体棒, "revolute" で折れ曲がり可能
    """

    link_length: float = 0.06    # 1リンクあたり6cm
    link_radius: float = 0.008   # 半径8mm
    joint_type: str = "fixed"    # "fixed" or "revolute"

    # --- ArticulationCfg の必須フィールドをデフォルト設定 ---
    prim_path: str = "{ENV_REGEX_NS}/Object"

    spawn: sim_utils.UsdFileCfg = MISSING  # __post_init__ で差し替える

    init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg(
        pos=(0.5, 0.0, 0.055),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={"rod_joint_1": 0.0, "rod_joint_2": 0.0},
        joint_vel={"rod_joint_1": 0.0, "rod_joint_2": 0.0},
    )

    actuators: dict = None  # __post_init__ で設定

    def __post_init__(self):
        import os, tempfile
        usd_path = self._generate_linked_rod_usd()

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

        damping = 0.5 if self.joint_type == "revolute" else 1e6
        stiffness = 0.0 if self.joint_type == "revolute" else 1e6
        self.actuators = {
            "rod_joints": ImplicitActuatorCfg(
                joint_names_expr=["rod_joint_.*"],
                stiffness=stiffness,
                damping=damping,
            )
        }

    def _generate_linked_rod_usd(self) -> str:
        """
        pxr (OpenUSD) を使って3リンク棒のUSDをテンポラリファイルに生成して返す。
        """
        import tempfile, os
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf

        usd_path = os.path.join(tempfile.gettempdir(), "linked_rod_3links.usd")

        stage = Usd.Stage.CreateNew(usd_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        L = self.link_length
        R = self.link_radius

        root_path = "/LinkedRod"
        root_xform = UsdGeom.Xform.Define(stage, root_path)

        # ArticulationRoot
        phys_root = UsdPhysics.ArticulationRootAPI.Apply(root_xform.GetPrim())

        def _make_link(name: str, parent_path: str, local_pos: tuple):
            link_path = f"{parent_path}/{name}"
            xform = UsdGeom.Xform.Define(stage, link_path)
            xform.AddTranslateOp().Set(Gf.Vec3f(*local_pos))

            # 円柱コリジョン
            cyl_path = f"{link_path}/CylinderCollision"
            cyl = UsdGeom.Cylinder.Define(stage, cyl_path)
            cyl.GetRadiusAttr().Set(R)
            cyl.GetHeightAttr().Set(L)
            UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

            # RigidBody
            rb = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())

            # Mass
            mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
            mass_api.GetMassAttr().Set(0.05 / 3.0)

            return link_path

        def _make_joint(joint_name: str, parent_path: str, child_path: str,
                        local_pos_parent: tuple, local_pos_child: tuple,
                        joint_type: str):
            joint_path = f"{parent_path}/{joint_name}"
            if joint_type == "fixed":
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
            else:  # revolute (X軸回り)
                joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                joint.GetAxisAttr().Set("X")
                joint.GetLowerLimitAttr().Set(-90.0)
                joint.GetUpperLimitAttr().Set(90.0)

            joint.GetBody0Rel().SetTargets([Sdf.Path(parent_path)])
            joint.GetBody1Rel().SetTargets([Sdf.Path(child_path)])
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*local_pos_parent))
            joint.GetLocalPos1Attr().Set(Gf.Vec3f(*local_pos_child))

        # link_0 (root): Z中心に配置
        link0 = _make_link("rod_link_0", root_path, (0.0, 0.0, 0.0))
        # link_1: link_0の上端に接続
        link1 = _make_link("rod_link_1", root_path, (0.0, 0.0, L))
        # link_2: link_1の上端に接続
        link2 = _make_link("rod_link_2", root_path, (0.0, 0.0, L * 2))

        # joint_1: link0 → link1
        _make_joint(
            "rod_joint_1",
            link0, link1,
            local_pos_parent=(0.0, 0.0,  L / 2.0),   # link0の上端
            local_pos_child= (0.0, 0.0, -L / 2.0),   # link1の下端
            joint_type=self.joint_type,
        )
        # joint_2: link1 → link2
        _make_joint(
            "rod_joint_2",
            link1, link2,
            local_pos_parent=(0.0, 0.0,  L / 2.0),
            local_pos_child= (0.0, 0.0, -L / 2.0),
            joint_type=self.joint_type,
        )

        stage.SetDefaultPrim(root_xform.GetPrim())
        stage.Save()
        return usd_path