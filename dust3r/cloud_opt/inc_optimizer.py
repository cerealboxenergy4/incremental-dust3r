# pseudo-code / safe to adapt

class IncrementalAligner:
    def __init__(self, dust3r_output, device, base_lr=1e-2, niter_boot=100, niter_step=50):
        self.output = dust3r_output
        self.device = device
        self.base_lr = base_lr
        self.niter_boot = niter_boot
        self.niter_step = niter_step

        # 전체 이미지 개수, 전체 pairs를 미리 파악
        self.view1, self.view2 = self.output['view1'], self.output['view2']
        self.pred1, self.pred2 = self.output['pred1'], self.output['pred2']

        # 내부 상태(누적) 저장소
        self.saved_poses = {}     # i -> 4x4 cam2world
        self.saved_focals = {}    # i -> f (또는 [fx, fy])
        self.saved_pp = {}        # i -> (cx, cy)
        self.saved_depth = {}     # i -> (Hi, Wi) torch.Tensor (depth)

    def _slice_pairs(self, allowed_edges):
        # allowed_edges: list of (i, j)
        # dust3r_output을 동일 포맷으로 잘라 반환
        # view1/view2/pred1/pred2 각 리스트를 같은 인덱싱으로 슬라이스
        keep = []
        for idx, (i, j) in enumerate(zip(self.view1['idx'], self.view2['idx'])):
            if (int(i), int(j)) in allowed_edges:
                keep.append(idx)

        def pick(d, keys):
            return {k: [d[k][n] for n in keep] for k in keys}

        sliced = {
            'view1': pick(self.view1, ['idx','img','true_shape','instance']),
            'view2': pick(self.view2, ['idx','img','true_shape','instance']),
            'pred1': pick(self.pred1, ['pts3d','conf']),
            'pred2': pick(self.pred2, ['pts3d_in_other_view','conf'])
        }
        return sliced

    def _build_optimizer(self, sliced, optimize_pp=False, fx_and_fy=False):
        # 반드시 ModularPointCloudOptimizer 사용 (부분 동결 지원)
        from dust3r.cloud_opt import ModularPointCloudOptimizer
        net = ModularPointCloudOptimizer(
            sliced['view1'], sliced['view2'], sliced['pred1'], sliced['pred2'],
            optimize_pp=optimize_pp, fx_and_fy=fx_and_fy
        ).to(self.device)
        return net

    def _freeze_old_and_inject(self, net, known_ids):
        # 포즈/내참수 preset API로 동결
        # (1) pose
        if len(known_ids) > 0:
            known_poses = [self.saved_poses[i] for i in known_ids]
            net.preset_pose(known_poses, pose_msk=known_ids)     # requires_grad False로 바뀜
        # (2) intrinsics
        if len(known_ids) > 0 and len(self.saved_focals) == len(self.saved_pp):
            known_K = []
            for i in known_ids:
                # K 구성: fx=fy=f, cx,cy = saved_pp
                import torch
                K = torch.zeros(3,3)
                f = self.saved_focals[i]
                cx, cy = self.saved_pp[i]
                K[0,0]=f; K[1,1]=f; K[0,2]=cx; K[1,2]=cy; K[2,2]=1
                known_K.append(K)
            net.preset_intrinsics(known_K, msk=known_ids)        # fx/fy/pp 주입 + 동결

        # (3) depth (내부 메서드로 주입 + 동결)
        for i in known_ids:
            if i in self.saved_depth:
                net._set_depthmap(i, self.saved_depth[i], force=True)  # log(depth)로 복사
                net.im_depthmaps[i].requires_grad_(False)               # 명시 동결

    def _run(self, net, niter, lr):
        # net.compute_global_alignment()를 직접 돌리거나,
        # 간단히 Adam 루프를 직접 작성해도 된다. 여기서는 내부 루틴 사용 가정.
        loss = net.compute_global_alignment(init=None, niter_PnP=0, niter=niter,
                                            schedule='cosine', lr=lr)   # 내부 루프
        return loss

    def _save_current_params(self, net):
        # 현재까지 학습된 파라미터를 saved_*에 복사
        import torch
        poses = net.get_im_poses().detach().cpu()
        focals = net.get_focals().detach().cpu()
        pps = net.get_principal_points().detach().cpu()
        depths = net.get_depthmaps()  # Modular: list of tensors (exp된 depth)
        for i in range(net.n_imgs):
            self.saved_poses[i] = poses[i]
            self.saved_focals[i] = float(focals[i].mean())  # fx=fy 가정
            self.saved_pp[i] = (float(pps[i,0]), float(pps[i,1]))
            self.saved_depth[i] = depths[i].detach().cpu()

    def bootstrap_first3(self):
        # order: 이미지 인덱스 등장 순서 (예: [0,1,2,3,4,...])
        # pairs: base_ids로만 구성된 모든 방향 or 원하는 그래프 생성
        allowed = []
        for a in range(0,3):
            for b in range(0,3):
                if a!=b: allowed.append((a,b))

        sliced = self._slice_pairs(allowed)
        net = self._build_optimizer(sliced)
        loss = self._run(net, self.niter_boot, self.base_lr)
        self._save_current_params(net)

    def step_add(self, order, k, recent_k=2):
        # 새 이미지 idx = order[k] 추가
        cur_ids = order[:k+1]
        new_id = order[k]
        prev_ids = order[:k]
        hook_ids = prev_ids[-recent_k:]  # 최근 2개
        # 신규 엣지: (new, hook) * 양방향
        allowed = []
        for h in hook_ids:
            allowed += [(new_id,h),(h,new_id)]

        # 또한, 이미 등록된 이미지들의 포즈는 쓰되 엣지는 만들지 않음
        sliced = self._slice_pairs(allowed)
        net = self._build_optimizer(sliced)

        # (중요) net은 지금 n_imgs가 "allowed edges로 등장한 idx 집합"의 크기임.
        # modular optimizer는 이미지 인덱스를 원본과 동일하게 유지한다.
        # 기존 것 동결 + 파라미터 주입
        self._freeze_old_and_inject(net, known_ids=prev_ids)

        # 이제 새 이미지 new_id만 학습된다(나머지는 requires_grad False).
        loss = self._run(net, self.niter_step, self.base_lr)

        # 업데이트된 전체 상태를 보존 (특히 new_id)
        self._save_current_params(net)
        return loss
