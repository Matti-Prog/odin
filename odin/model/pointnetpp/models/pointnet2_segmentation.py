import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import square_distance, index_points, PointNetSetAbstractionMsg

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)), inplace=True)
        return new_points

class PointNetv2Seg(nn.Module):
    def __init__(
        self, num_class=24, in_channel=None, out_dim=128,
        sa1_params=None, sa2_params=None, sa3_params=None, sa4_params=None,
        fp1_params=None, fp2_params=None, fp3_params=None, fp4_params=None,
        checkpoint_path = None
    ):

        super(PointNetv2Seg, self).__init__()
        self.in_channel = 3 if in_channel is None else in_channel
        if sa1_params is None:
            sa1_params = dict()
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=sa1_params.get("npoint", 1024),
            radius_list=sa1_params.get("radius_list", [0.05, 0.1]),
            nsample_list=sa1_params.get("nsample_list", [16, 32]),
            in_channel=self.in_channel,
            mlp_list=sa1_params.get("mlp_list", [[16, 16, 32], [32, 32, 64]])
        )
        if sa2_params is None:
            sa2_params = dict()
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=sa2_params.get("npoint", 256),
            radius_list=sa2_params.get("radius_list", [0.1, 0.2]),
            nsample_list=sa2_params.get("nsample_list", [16, 32]),
            in_channel=sa2_params.get("in_channel", 96),
            mlp_list=sa2_params.get("mlp_list", [[64, 64, 128], [64, 96, 128]])
        )
        if sa3_params is None:
            sa3_params = dict()
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=sa3_params.get("npoint", 64),
            radius_list=sa3_params.get("radius_list", [0.2, 0.4]),
            nsample_list=sa3_params.get("nsample_list", [16, 32]),
            in_channel=sa3_params.get("in_channel", 256),
            mlp_list=sa3_params.get("mlp_list", [[128, 196, 256], [128, 196, 256]])
        )
        if sa4_params is None:
            sa4_params = dict()
        self.sa4 = PointNetSetAbstractionMsg(
            npoint=sa4_params.get("npoint", 1),
            radius_list=sa4_params.get("radius_list", [0.4, 0.8]),
            nsample_list=sa4_params.get("nsample_list", [16, 32]),
            in_channel=sa4_params.get("in_channel", 512),
            mlp_list=sa4_params.get("mlp_list", [[128, 128], [128, 128]])
        )
        if fp4_params is None:
            fp4_params = dict()
        self.fp4 = PointNetFeaturePropagation(
            in_channel=fp4_params.get("in_channel", 512 + 256),
            mlp=fp4_params.get("mlp", [256, 256])
        )
        if fp3_params is None:
            fp3_params = dict()
        self.fp3 = PointNetFeaturePropagation(
            in_channel=fp3_params.get("in_channel", 128 + 128 + 256),
            mlp=fp3_params.get("mlp", [256, 256])
        )
        if fp2_params is None:
            fp2_params = dict()
        self.fp2 = PointNetFeaturePropagation(
            in_channel=fp2_params.get("in_channel", 32 + 64 + 256),
            mlp=fp2_params.get("mlp", [256, 128])
        )
        if fp1_params is None:
            fp1_params = dict()
        self.fp1 = PointNetFeaturePropagation(
            in_channel=fp1_params.get("in_channel", 128),
            mlp=fp1_params.get("mlp", [128, 128, 128])
        )
        self.decoder = self.create_decoder(out_dim, num_class)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        self.eval()

    @staticmethod
    def create_decoder(in_dim, out_dim):
        return nn.Sequential(
            nn.Conv1d(in_dim, in_dim, 1),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_dim, out_dim, 1)
        )

    def forward(self, xyz):
        xyz = xyz.transpose(2, 1)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        global_feature = l4_points.squeeze(-1)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        local_feature = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        segmentation_logits = self.decoder(local_feature)
        segmentation_logits = segmentation_logits.permute(0, 2, 1)
        segmentation_logits = F.log_softmax(segmentation_logits, dim=2)
        segmentation = torch.argmax(segmentation_logits, dim=2)
        return global_feature, local_feature, segmentation