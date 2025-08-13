# incremental_optimizer.py
import numpy as np
import torch
from dust3r.cloud_opt.modular_optimizer import ModularPointCloudOptimizer
from dust3r.cloud_opt.commons import edge_str
from dust3r.utils.geometry import geotrf
from dust3r.cloud_opt.init_im_poses import init_from_known_poses


class IncrementalPCOptimizer(ModularPointCloudOptimizer):
    """
    Incremental global alignment:
      - keep full dataset loaded, but optimize loss only on self.active_edges
      - freeze already-registered images; update newly-added image(s) only
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # loads ALL edges/images, builds pred_i/j dicts, imshapes ...
        # active nodes/edges (start empty; user calls bootstrap() to seed)
        self.active_nodes = set()
        self.active_edges = []  # list of tuples (i, j)
        # mapping from an edge to its global index 'e' (for pw_poses/adaptors)
        self.edge_to_idx = {edge_str(i, j): e for e, (i, j) in enumerate(self.edges)}

    # ---------- helpers: freeze/thaw images ----------
    def freeze_images(self, msk):
        """msk: int | list[int] | np.array(bool/int)"""
        idxs = self._get_msk_indices(msk)         # from ModularPointCloudOptimizer
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
    def bootstrap(self, seed_ids=[0,1,2], fully_connect=True):
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
                self._maybe_add_edge((a, b)); self._maybe_add_edge((b, a))

        # thaw seeds for initial solve, freeze the rest
        self.thaw_images(seeds, optimize_pp=False)
        frozen = [i for i in range(self.n_imgs) if i not in self.active_nodes]
        if len(frozen):
            self.freeze_images(frozen)

    @torch.no_grad()
    def add_image_with_hooks(self, new_id, hook_ids, optimize_pp=True):
        # 0) 엣지 등록 (new↔hooks)
        added = []
        for h in hook_ids:
            if self._maybe_add_edge((new_id, h)): added.append((new_id, h))
            if self._maybe_add_edge((h, new_id)): added.append((h, new_id))

        # 1) 전부 잠깐 freeze (init_from_known_poses가 '모두 known'을 요구)
        all_ids = list(range(self.n_imgs))
        self.freeze_images(all_ids)  # im_poses / im_depth / focals / pp 다 freeze 상태로

        # 2) 새 프레임 포즈/내참수 초기값 주입
        self.init_pose_from_prev(new_id, hook_ids, mode="copy")  # mode = "copy" or "velocity"
        #   intrinsics는 훅에서 복사(반드시 CPU로 건네기)
        K0 = self.get_intrinsics()[hook_ids[0]].detach().cpu()
        self.preset_intrinsics([K0], msk=[new_id])  # f, pp 주입 + 동결

        # 3) 한 방에 초기화: pw_poses + depthmaps
        init_from_known_poses(self, niter_PnP=10)   # ← 여기!
        #   - 각 엣지(e)의 pw_pose를 PnP/정합으로 세팅
        #   - 각 이미지의 depth를 pred_i[...,2]*scale로 세팅

        # 4) 이제 새 프레임만 thaw해서 학습
        self.freeze_images(all_ids)                      # 일단 다시 전부 freeze
        self.thaw_images([new_id], optimize_pp=optimize_pp)
        # (원하면 pw_poses도 활성화된 엣지만 학습되도록 그대로 두면 됨)
        """
        Add one new image and connect it only to hook_ids (e.g., the most recent 2)
        hook_ids: iterable (e.g., [k-1, k-2])
        
        self.active_nodes.add(new_id)
        for h in hook_ids:
            self._maybe_add_edge((new_id, h))
            self._maybe_add_edge((h, new_id))

        # freeze all existing; thaw the new image
        self.freeze_images(list(self.active_nodes - {new_id}))
        self.thaw_images([new_id], optimize_pp=optimize_pp)

        self.init_pose_from_prev(new_id, hook_ids, mode="copy")  # or "copy"

        # ▶▶ 여기! Intrinsics 복사 + 동결
        hook = hook_ids[0]                     # 보통 k-1
        K0   = self.get_intrinsics()[hook].detach().cpu()
        self.preset_intrinsics([K0], msk=[new_id])   # f, pp 주입 + requires_grad=False

        from dust3r.cloud_opt.commons import edge_str

        # cam2world(i)값으로 pw_poses 초기화
        with torch.no_grad():
            Ti_all = self.get_im_poses().detach()
            for h in hook_ids:
                for (i, j) in [(new_id, h), (h, new_id)]:
                    key = edge_str(i, j)
                    e = self.edge_to_idx[key]
                    self._set_pose(self.pw_poses, e, Ti_all[i], force=True)  # pw_pose ≈ cam2world(i)

             # hooks에서 온 DUSt3R 예측 z들을 모아서 중간값을 초기 깊이로
            # zs = []
            # for h in hook_ids:
            #     key = edge_str(new_id, h)
            #     z = self.pred_i[key][..., 2].reshape(-1)      # i=new 좌표의 z
            #     zs.append(z[torch.isfinite(z)])
            # z0 = torch.median(torch.cat(zs)) if len(zs) else torch.tensor(1.0, device=self.device)

            # H, W = self.imshapes[new_id]
            # init_depth = z0.expand(H, W)                      # 간단한 평탄 초기화
            # self._set_depthmap(new_id, init_depth, force=True)  # log로 저장됨  :contentReference[oaicite:7]{index=7}
            """

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
            T_init  = Tkm1 @ T_delta

        # >>> 핵심: 내부에서 roma.rotmat_to_unitquat() 호출해서 pose 파라미터에 반영
        self._set_pose(self.im_poses, new_id, T_init, force=True)  # 4x4 전달 OK :contentReference[oaicite:2]{index=2}


    # ---------- override forward to use only active_edges ----------
    def forward(self, ret_details=False):
        # pairwise RT & adaptors
        pw_poses = self.get_pw_poses()          # (n_edges,4,4); global list, we'll index by edge_to_idx  
        pw_adapt = self.get_adaptors()          # (n_edges,3)    3-axis scale factors                   
        proj_pts3d = self.get_pts3d()           # per-image world pointmaps

        # weights per edge
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}  # like BasePCOptimizer.forward  
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0.0
        if ret_details:
            details = -torch.ones((self.n_imgs, self.n_imgs), device=self.device)

        for (i, j) in self.active_edges:
            key = edge_str(i, j)
            e = self.edge_to_idx[key]

            aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[key])  # 3D align i->*
            aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[key])  # 3D align j->*
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
    
    def compute_incremental_alignment(
    self,
    order,                # 정렬에 쓸 이미지 인덱스 순서 (예: [0,1,2,3,...])
    seed=3,               # 처음 부트스트랩에 쓸 개수
    hooks=2,              # 새 이미지가 연결할 최근 이웃 개수
    init="mst",           # 부트스트랩 정렬 초기화 ("mst"/None/등)
    niter_boot=300,       # 부트스트랩 최적화 횟수
    niter_step=100,        # 증분 스텝 최적화 횟수
    schedule="cosine",
    lr_boot=1e-2,
    lr_step=5e-3,
    optimize_pp=False,    # 새 이미지에서 pp까지 풀지 여부
):
        """
        증분 전 과정을 한 번에 실행하고 마지막 loss(float)를 반환.
        BasePCOptimizer.compute_global_alignment(...)를 내부에서 그대로 재사용.
        """
        # 1) 부트스트랩
        seeds = list(order[:seed])
        self.bootstrap(seed_ids=seeds, fully_connect=True)
        self.compute_global_alignment(   # BasePCOptimizer가 제공하는 동일 API 재사용
            init=init, niter=niter_boot, schedule=schedule, lr=lr_boot
        )

        # 2) 증분 스텝
        for k in range(seed, len(order)):
            new_id = int(order[k])
            hook_ids = [int(h) for h in order[max(0, k - hooks):k]]
            self.add_image_with_hooks(new_id=new_id, hook_ids=hook_ids, optimize_pp=optimize_pp)
            self.compute_global_alignment(
                init=init, niter=niter_step, schedule=schedule, lr=lr_step
            )

        # compute_global_alignment는 내부에서 loss를 반환하므로 마지막 것을 그대로 리턴
        return self.forward().item()
