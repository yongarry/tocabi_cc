import torch

class preview_controller:
    def __init__(self, num_envs: int, device: torch.device, time_horizon: float = 2, dt: float = 0.008):
        self.num_envs = num_envs
        self.device = device
        self.time_horizon = time_horizon
        self.dt = dt
        self.hz = int(1/dt)
        self.NL = int(time_horizon*self.hz)

        X_dare = torch.zeros((self.NL,4,1), device=self.device)
        X_dare[0] = torch.tensor([  -69.0813009391229,
                                    -2420.65372019103,
                                    -669.387885399942,
                                    -2.72127899669319], device=self.device).reshape(4, 1)
        ## this parameter is tuned for com height: 0.728m
        self.G_d_dare = torch.zeros((self.NL,), device=self.device)

        self.K_dare = torch.tensor([[70.0813009391222,	2420.65372019101,	669.387885399937,	2.72127899669319],
                                    [2420.65372019101,	85969.0761591658,	23782.1340369413,	99.0855302680049],
                                    [669.387885399937,	23782.1340369413,	6579.12894080612,	27.4486400273521],
                                    [2.72127899669319,	99.0855302680049,	27.4486400273521,	0.125016968799296]],    
                                    device=self.device).unsqueeze(dim=0)
        
        self.G_i_dare = 556.382091536873
        self.G_x_dare = torch.tensor([[38991.9807941560,	11086.4028841705	,130.234568101964]], device=self.device).unsqueeze(dim=0)
        self.G_d_dare[0] = -self.G_i_dare

        Ac_bar_T = torch.tensor([[1.33026539679257,	-4.74779384778249e-05,	-0.0178042269291843,	-4.45105673229607],
                                 [24.1454286624088	,0.996672684305565,	-1.24774338541315	,-311.935846353287],
                                 [6.58882872047239,	0.00705396028721734,	0.645235107706503,	-88.6912230733744],
                                 [0.00312854461408964,	2.08866501886319e-05,	0.00383249382073698,	-0.0418765448157545]], 
                                device=self.device)
        RBT = torch.tensor([[-183.753752229875,	0.0264158747121758	,9.90595301706591	,2476.48825426648]], device=self.device)

        for l in torch.arange(start=1, end=self.NL):
            X_dare[l] = torch.mm(Ac_bar_T, X_dare[l-1])
            self.G_d_dare[l] = torch.mm(RBT, X_dare[l-1]).squeeze().squeeze()

        self.A = torch.tensor([[1, self.dt, pow(self.dt, 2)/2],
                               [0, 1, self.dt],
                               [0, 0, 1]], device=self.device).unsqueeze(dim=0)
        self.B = torch.tensor([[pow(self.dt,3)/6], [pow(self.dt, 2)/2], [self.dt]], device=self.device).unsqueeze(dim=0)
        self.C = torch.tensor([[1, 0, -0.728/9.81]], device=self.device).unsqueeze(dim=0)

        self.error_integral = torch.zeros((self.num_envs, 3), device=self.device)

        self.state = torch.zeros((self.num_envs, 3, 3), device=self.device)

    def compute_target_state(self, vrp_ref: torch.Tensor):
        with torch.no_grad():
            assert vrp_ref.shape == (3, self.num_envs, self.NL),  f"Given vrp ref shape : {vrp_ref.shape}"
            # 3, num_envs, NL -> NL, num_envs, 3
            vrp_ref_permute = vrp_ref.permute(2, 1, 0)

            self.error_integral += torch.bmm(self.C.repeat((self.num_envs, 1, 1)), self.state).squeeze() - vrp_ref_permute[0, :]

            input = -self.G_i_dare * self.error_integral \
                    -torch.bmm(self.G_x_dare.repeat((self.num_envs, 1, 1)), self.state).squeeze(dim=1) \
                    -(self.G_d_dare[1:self.NL].reshape(-1, 1, 1).repeat(1, self.num_envs, 3) * vrp_ref_permute[1:self.NL]).sum(dim=0)
            assert input.shape == (self.num_envs, 3), f"output shape is wrong, {input.shape}"
            
            return torch.bmm(self.A.repeat((self.num_envs, 1, 1)), self.state) + torch.bmm(self.B.repeat((self.num_envs, 1, 1)), input.unsqueeze(dim=1))

    def update_state(self, next_state: torch.Tensor):
        assert next_state.shape == (self.num_envs, 3, 3), f"next_state shape is wrong, {next_state.shape}"
        self.state = next_state.clone()