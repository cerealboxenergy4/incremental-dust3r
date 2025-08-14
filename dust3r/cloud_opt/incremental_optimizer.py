# incremental_optimizer.py
import numpy as np
import torch
from dust3r.cloud_opt.modular_optimizer import ModularPointCloudOptimizer
from dust3r.cloud_opt.commons import edge_str
from dust3r.utils.geometry import geotrf
from dust3r.cloud_opt.init_im_poses import init_from_known_poses
from dust3r.utils.image import rgb


class IncrementalPCOptimizer(ModularPointCloudOptimizer):
    """
    Incremental global alignment:
      - keep full dataset loaded, but optimize loss only on self.active_edges
      - freeze already-registered images; update newly-added image(s) only
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )  # loads ALL edges/images, builds pred_i/j dicts, imshapes ...
        # active nodes/edges (start empty; user calls bootstrap() to seed)
        self.active_nodes = set()
        self.active_edges = []  # list of tuples (i, j)
        # mapping from an edge to its global index 'e' (for pw_poses/adaptors)
        self.edge_to_idx = {edge_str(i, j): e for e, (i, j) in enumerate(self.edges)}

    # ---------- helpers: freeze/thaw images ----------
    def freeze_images(self, msk):
        """msk: int | list[int] | np.array(bool/int)"""
        idxs = self._get_msk_indices(msk)  # from ModularPointCloudOptimizer
        # pose
        for i in idxs:
            self.im_poses[i].requires_grad_(False)
        # intrinsics
        for i in idxs:
            self.im_focals[i].requires_grad_(False)
            self.im_pp[i].requires_grad_(False)
        # depth
        for i in idxs:
            self.im_depthmaps[i].requires_grad_(False)

    def thaw_images(self, msk, optimize_pp=False):
        idxs = self._get_msk_indices(msk)
        for i in idxs:
            self.im_poses[i].requires_grad_(True)
            self.im_depthmaps[i].requires_grad_(True)
            self.im_focals[i].requires_grad_(True)
            self.im_pp[i].requires_grad_(optimize_pp)

    # ---------- incremental API ----------
    @torch.no_grad()
    def bootstrap(self, seed_ids=[0, 1, 2], fully_connect=True):
        """
        seed_ids: iterable of image indices to initialize (e.g., first 3)
        fully_connect=True -> use all directed edges among seeds; else only consecutive pairs
        """
        self.active_nodes = set(seed_ids)
        self.active_edges = []
        seeds = list(seed_ids)
        if fully_connect:
            for a in seeds:
                for b in seeds:
                    if a != b:
                        self._maybe_add_edge((a, b))
        else:
            for a, b in zip(seeds[:-1], seeds[1:]):
                self._maybe_add_edge((a, b))
                self._maybe_add_edge((b, a))

        # thaw seeds for initial solve, freeze the rest
        self.thaw_images(seeds, optimize_pp=False)
        frozen = [i for i in range(self.n_imgs) if i not in self.active_nodes]
        if len(frozen):
            self.freeze_images(frozen)

    @torch.no_grad()
    def add_image_with_hooks(self, new_id, hook_ids, optimize_pp=True): # unused by the moment
        # 0) 엣지 등록 (new↔hooks)
        added = []
        for h in hook_ids:
            if self._maybe_add_edge((new_id, h)):
                added.append((new_id, h))
            if self._maybe_add_edge((h, new_id)):
                added.append((h, new_id))

        # 1) 전부 잠깐 freeze (init_from_known_poses가 '모두 known'을 요구)
        all_ids = list(range(self.n_imgs))
        self.freeze_images(
            all_ids
        )  # im_poses / im_depth / focals / pp 다 freeze 상태로

        # 2) 새 프레임 포즈/내참수 초기값 주입
        self.init_pose_from_prev(
            new_id, hook_ids, mode="copy"
        )  # mode = "copy" or "velocity"
        #   intrinsics는 훅에서 복사(반드시 CPU로 건네기)
        K0 = self.get_intrinsics()[hook_ids[0]].detach().cpu()
        self.preset_intrinsics([K0], msk=[new_id])  # f, pp 주입 + 동결

        # 3) 한 방에 초기화: pw_poses + depthmaps
        init_from_known_poses(self, niter_PnP=10)  # ← 여기!
        #   - 각 엣지(e)의 pw_pose를 PnP/정합으로 세팅
        #   - 각 이미지의 depth를 pred_i[...,2]*scale로 세팅

        # 4) 이제 새 프레임만 thaw해서 학습
        self.freeze_images(all_ids)  # 일단 다시 전부 freeze
        self.thaw_images([new_id], optimize_pp=optimize_pp)
        # (원하면 pw_poses도 활성화된 엣지만 학습되도록 그대로 두면 됨)

    @torch.no_grad()
    def add_image_incrementally(
        self, idx, hooks, added_view1, added_view2, added_pred1, added_pred2
    ):
        # 0) 새 엣지 구성
        edges_new = [
            (int(i), int(j)) for i, j in zip(added_view1["idx"], added_view2["idx"])
        ]

        # 1) edges / n_imgs 갱신
        self.edges += edges_new
        self.n_imgs = self._check_edges()  # 0..N-1 연속성 보장


        # img 추가
        self.imgs.extend([None] * (self.n_imgs - len(self.imgs)))
        if 'img' in added_view1 and 'img' in added_view2:
            for v in range(len(edges_new)):
                idx = added_view1['idx'][v]
                self.imgs[idx] = rgb(added_view1['img'][v])
                idx = added_view2['idx'][v]
                self.imgs[idx] = rgb(added_view2['img'][v])


        # 2) pred/conf 사전에 주입
        for n, (i, j) in enumerate(edges_new):
            k = edge_str(i, j)
            self.pred_i[k] = torch.nn.Parameter(
                added_pred1["pts3d"][n].to(self.device).detach().clone(),
                requires_grad=False,
            )
            self.pred_j[k] = torch.nn.Parameter(
                added_pred2["pts3d_in_other_view"][n].to(self.device).detach().clone(),
                requires_grad=False,
            )
            self.conf_i[k] = torch.nn.Parameter(
                added_pred1["conf"][n].to(self.device).detach().clone(),
                requires_grad=False,
            )
            self.conf_j[k] = torch.nn.Parameter(
                added_pred2["conf"][n].to(self.device).detach().clone(),
                requires_grad=False,
            )

        # 3) 새 이미지면 파라미터 append (idx는 반드시 기존의 최댓값+1 권장)
        if idx == max([i for e in self.edges for i in e]):
            H, W, _ = self.pred_i[edge_str(idx, added_view2["idx"][0])].shape
            self.imshapes.append((H, W))

            # depth / pose / focal / pp 파라미터 추가
            self.im_depthmaps.append(
                torch.nn.Parameter(torch.randn(H, W, device=self.device) / 10 - 3)
            )  # log-depth
            self.im_poses.append(self.rand_pose(self.POSE_DIM).to(self.device))
            self.init_pose_from_prev(
                idx, [idx - i - 1 for i in range(hooks)], mode="copy"
            )

            default_f = self.focal_brake * np.log(max(H, W))
            self.im_focals.append(
                torch.nn.Parameter(
                    torch.tensor([default_f], device=self.device).float()
                )
            )
            self.im_pp.append(torch.nn.Parameter(torch.zeros(2, device=self.device)))

            K0 = self.get_intrinsics()[idx - 1].detach().cpu()
            self.preset_intrinsics([K0], msk=[idx])  # f, pp 주입 + 동결

            # im_conf 엔트리도 하나 추가(zeros) 후, 아래 4)에서 max로 갱신
            self.im_conf.append(torch.zeros((H, W), device=self.device))

        # 4) im_conf 업데이트(새로 들어온 엣지의 conf로 max-accumulate)
        for i, j in edges_new:
            k = edge_str(i, j)
            self.im_conf[i] = torch.nn.Parameter(torch.maximum(self.im_conf[i], self.conf_i[k]), requires_grad=False)
            self.im_conf[j] = torch.nn.Parameter(torch.maximum(self.im_conf[j], self.conf_j[k]), requires_grad=False)

        # 5) pw_poses / pw_adaptors 확장 + 초기화
        #    (간단히: 각 새 엣지 e=(i,j)의 pw_pose를 cam2world(i)로 시작)
        old_E = self.pw_poses.shape[0]
        add_E = len(edges_new)
        new_pw = torch.zeros((old_E + add_E, 1 + self.POSE_DIM), device=self.device)
        new_ad = torch.zeros((old_E + add_E, 2), device=self.device)
        new_pw[:old_E] = self.pw_poses.data
        new_ad[:old_E] = self.pw_adaptors.data
        self.pw_poses = torch.nn.Parameter(new_pw)
        self.pw_adaptors = torch.nn.Parameter(new_ad)

        Ti_all = self.get_im_poses().detach()
        for t, (i, j) in enumerate(edges_new, start=old_E):
            self._set_pose(self.pw_poses, t, Ti_all[i], force=True)  # R,T,scale 세팅

        all_ids = list(range(self.n_imgs))
        self.freeze_images(all_ids)
        self.thaw_images([idx], optimize_pp=False)

        self.edge_to_idx = {edge_str(i, j): e for e, (i, j) in enumerate(self.edges)}

    def _maybe_add_edge(self, edge):
        i, j = edge
        key = edge_str(i, j)
        if key not in self.edge_to_idx:
            # the pair does not exist in DUSt3R outputs; skip safely
            return False
        if (i, j) not in self.active_edges:
            self.active_edges.append((i, j))
        return True

    @torch.no_grad()
    def init_pose_from_prev(self, new_id: int, hooks, mode="copy"):
        Tkm1 = self.get_im_poses()[hooks[-1]]  # (4,4) cam->world
        if mode == "copy" or len(hooks) == 1:
            T_init = Tkm1
        else:  # "velocity": T_k ≈ T_{k-1} * (T_{k-2}^{-1} T_{k-1})
            Tkm2 = self.get_im_poses()[hooks[-2]]
            T_delta = torch.linalg.inv(Tkm2) @ Tkm1
            T_init = Tkm1 @ T_delta

        # >>> 핵심: 내부에서 roma.rotmat_to_unitquat() 호출해서 pose 파라미터에 반영
        self._set_pose(
            self.im_poses, new_id, T_init, force=True
        )  # 4x4 전달 OK :contentReference[oaicite:2]{index=2}

    # ---------- override forward to use only active_edges ----------
    def forward(self, ret_details=False):
        # pairwise RT & adaptors
        pw_poses = (
            self.get_pw_poses()
        )  # (n_edges,4,4); global list, we'll index by edge_to_idx
        pw_adapt = self.get_adaptors()  # (n_edges,3)    3-axis scale factors
        proj_pts3d = self.get_pts3d()  # per-image world pointmaps

        # weights per edge
        weight_i = {
            i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()
        }  # like BasePCOptimizer.forward
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0.0
        if ret_details:
            details = -torch.ones((self.n_imgs, self.n_imgs), device=self.device)

        for i, j in self.active_edges:
            key = edge_str(i, j)
            e = self.edge_to_idx[key]

            aligned_pred_i = geotrf(
                pw_poses[e], pw_adapt[e] * self.pred_i[key]
            )  # 3D align i->*
            aligned_pred_j = geotrf(
                pw_poses[e], pw_adapt[e] * self.pred_j[key]
            )  # 3D align j->*
            li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[key]).mean()
            lj = self.dist(proj_pts3d[j], aligned_pred_j, weight=weight_j[key]).mean()
            loss = loss + li + lj

            if ret_details:
                details[i, j] = li + lj

        # normalize by the number of active edges (avoid div-by-zero)
        denom = max(len(self.active_edges), 1)
        loss = loss / denom

        if ret_details:
            return loss, details
        return loss

    def compute_one_step_alignment(
        self,
        niter_step=100,  # 증분 스텝 최적화 횟수
        schedule="linear",
        lr_step=1e-2,
    ):
        # assert len(self._get_msk_indices()) == 1

        self.compute_global_alignment(
            init="mst", niter=niter_step, schedule=schedule, lr=lr_step
        )
        return self.forward().item()

    def compute_initial_alignment(
        self,
        seed=3,
        niter=300,
        schedule="linear",
        lr=1e-2,
    ):
        seeds = list(range(seed))
        self.bootstrap(seed_ids=seeds, fully_connect=True)
        return self.compute_global_alignment(  # BasePCOptimizer가 제공하는 동일 API 재사용
            init="mst", niter=niter, schedule=schedule, lr=lr
        )
