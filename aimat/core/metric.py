from dataclasses import dataclass, field
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score
from numpy.core.numeric import NaN


@dataclass
class Metric:
    values: list[float] = field(default_factory=list)
    running_total: float = 0.0
    num_updates: float = 0.0
    average: float = 0.0

    def update(self, value: float, batch_size: int):
        self.values.append(value)
        self.running_total += value * batch_size
        self.num_updates += batch_size
        self.average = self.running_total / self.num_updates


class MetricMAE(Metric):
    is_np = False

    def calc(self, pred, true, n_batch):
        rst = (pred - true).abs().mean().item()
        self.update(rst, n_batch)


class MetricMSE(Metric):
    is_np = False

    def calc(self, pred, true, n_batch):
        rst = (pred - true).square().mean().item()
        self.update(rst, n_batch)


class MetricRMSE(Metric):
    is_np = False

    def calc(self, pred, true, n_batch):
        rst = (pred - true).square().mean().sqrt().item()
        self.update(rst, n_batch)


class MetricSSIM(Metric):
    is_np = True

    def calc(self, pred, true, n_batch):
        assert pred.ndim == 4
        assert true.ndim == 4

        rst = np.zeros(n_batch, dtype=np.float32)
        for idx_batch in range(n_batch):
            rst[idx_batch] = ssim(true[idx_batch], pred[idx_batch], channel_axis=0)

        rst = rst.mean()  # batch mean
        self.update(rst, n_batch)


class MetricPSNR(Metric):
    is_np = True

    def calc(self, pred, true, n_batch):
        rst = np.zeros(n_batch, dtype=np.float32)
        for idx_batch in range(n_batch):
            rst = psnr(true[idx_batch], pred[idx_batch])

        rst = rst.mean()  # batch mean
        self.update(rst, n_batch)


class MetricAccuracy(Metric):
    is_np = True

    def calc(self, pred, true, n_batch):
        rst = accuracy_score(true, pred)
        self.update(rst, n_batch)


class MetricF1:
    is_np = True

    def __init__(self):
        self.result_dict = dict()

    def calc(self, pred: np.ndarray, true: np.ndarray, n_batch: int, num_classes: int):
        """
        pred: shape[batch, H, W] or [batch]
        true: shape[batch, H, W] or [batch]
        """
        # The codes below are based on NamColorful's code ******************************************
        assert pred.shape == true.shape

        if len(true.shape) == 3:
            true = true[:len(pred), :len(pred[0]), :len(pred[0][0])]
        else:
            true = true[:len(pred), :len(pred[0])]
        labels = np.array(range(num_classes))

        pre = {}
        rec = {}

        for label in labels:
            true_positives = np.sum(np.logical_and(np.equal(true, label), np.equal(pred, label)))
            false_positives = np.sum(
                np.logical_and(np.logical_not(np.equal(true, label)), np.equal(pred, label))
            )
            false_negatives = np.sum(
                np.logical_and(np.equal(true, label), np.logical_not(np.equal(pred, label)))
            )

            if (true_positives + false_negatives) > 0:
                rec[f"Rec_{label}"] = true_positives / (true_positives + false_negatives)
            else:
                rec[f"Rec_{label}"] = NaN

            if (true_positives + false_positives) > 0:
                pre[f"Pre_{label}"] = true_positives / (true_positives + false_positives)
            else:
                pre[f"Pre_{label}"] = NaN

        mean_pre = np.nanmean(list(pre.values()))
        mean_rec = np.nanmean(list(rec.values()))
        f1_dict = {
            "f1": 2 * mean_pre * mean_rec/(mean_pre + mean_rec),
            "Pre": mean_pre,
            "Rec": mean_rec,
        }
        f1_dict.update(pre)
        f1_dict.update(rec)
        # ******************************************************************************************

        # 각각의 key에 대해 개별적으로 update
        for name_sub, value in f1_dict.items():
            if name_sub not in self.result_dict:
                self.result_dict[name_sub] = Metric()
            self.result_dict[name_sub].update(value, n_batch)


class MetricIOU:
    is_np = True

    def __init__(self):
        self.result_dict = dict()

    def calc(self, pred: np.ndarray, true: np.ndarray, n_batch: int, num_classes: int) -> dict:
        """
        pred: shape[batch, H, W] or [batch]
        true: shape[batch, H, W] or [batch]
        """
        # The codes below are based on NamColorful's code ******************************************
        assert pred.shape == true.shape

        if len(true.shape) == 3:
            true = true[:len(pred), :len(pred[0]), :len(pred[0][0])]
        else:
            true = true[:len(pred), :len(pred[0])]

        labels = np.array(range(num_classes))
        ious = []
        for label in labels:
            intersection = np.sum(np.logical_and(np.equal(true, label), np.equal(pred, label)))
            union = np.sum(np.logical_or(
                np.equal(true, label), np.equal(pred, label)))
            label_iou = intersection*1.0/union if union > 0 else NaN
            ious.append(label_iou)
        iou_dict = {"IOU_{}".format(label): iou for label, iou in zip(labels, ious)}
        iou_dict["mean_IOU_no_minority"] = np.nanmean(np.delete(ious, 3))  # class 3 is minority class in some datasets
        iou_dict["mean_IOU"] = np.nanmean(ious)
        # ******************************************************************************************

        # 각각의 key에 대해 개별적으로 update
        for name_sub, value in iou_dict.items():
            if name_sub not in self.result_dict:
                self.result_dict[name_sub] = Metric()
            self.result_dict[name_sub].update(value, n_batch)


class MetricCaller:
    """
    Runner 내의 학습 코드 안에서 instance가 생성되고 calc_metrics call 됨
    """
    def __init__(self, name_list: list) -> None:
        self.pred_np = None  # type casting 최소화 목적
        self.true_np = None

        self.name_list = name_list

        ''' NOTE:
        - user가 custom_callback 추가하는 경우 고려, self.calc_callback_dict은 instance attr여야 함.
        - 새로운 built-in callback(MetricMETRICNAME in metric.py) 추가한 경우 여기서 연결해줘야함
        '''
        self.callback_dict = {
            'mae': MetricMAE,
            'mse': MetricMSE,
            'rmse': MetricRMSE,
            'ssim': MetricSSIM,
            'psnr': MetricPSNR,
            'accuracy': MetricAccuracy,
            'f1': MetricF1,
            'IOU': MetricIOU,
        }

        # name_list element 처리할 수 있는지, 전달된 이름이 적절한지 검사
        for name_metric in self.name_list:
            assert name_metric in self.callback_dict

        self.result_total = {  # 실제 사용할 callback만 저장된 dict
            name_metric: self.callback_dict[name_metric]() for name_metric in self.name_list
        }

    def flush_data(self):  # 다음 batch data 넘어가기 전에 call 해줘야 하는 method
        self.pred_np = None
        self.true_np = None

    def calc_metrics(self, pred: torch.Tensor, true: torch.Tensor, idx_batch_dim=0, **kwargs):
        assert isinstance(pred, torch.Tensor)
        assert isinstance(true, torch.Tensor)
        assert pred.size() == true.size()  # pred와 true의 shape이 동일한지 검사
        n_batch = pred.size(idx_batch_dim)

        for name_metric in self.name_list:
            # (pred, true)가 numpy.ndarray 객체여야 계산 가능한 경우
            if self.result_total[name_metric].is_np:
                if self.pred_np is None:
                    # TODO ckChef: clone 필요한지? 실제 테스트에서 확인 필요
                    self.pred_np = pred.clone().cpu().detach().numpy()
                if self.true_np is None:
                    self.true_np = true.clone().cpu().detach().numpy()

                if name_metric in ('f1', 'IOU'):
                    self.result_total[name_metric].calc(
                        self.pred_np,
                        self.true_np,
                        n_batch,
                        **kwargs,
                    )
                else:
                    self.result_total[name_metric].calc(self.pred_np, self.true_np, n_batch)

            # torch.Tensor 객체 에서 바로 계산할 수 있는 경우
            else:
                self.result_total[name_metric].calc(pred, true, n_batch)

        self.flush_data()

    def get_avg_result(self):
        """
        - 
        """
        net_result = dict()
        for name_metric, callback in self.result_total.items():
            if isinstance(callback, Metric):
                net_result[name_metric] = callback.average
            else:  # output이 여러개라서 dict인 경우
                for name_metric_sub in callback.result_dict:
                    net_result[name_metric_sub] = callback.result_dict[name_metric_sub].average

        return net_result
