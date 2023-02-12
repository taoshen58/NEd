import logging

import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax


def judgment_distribution_to_score(judgment_distribution):
    n_class = judgment_distribution.shape[-1]
    weights = torch.arange(1.0, -0.01, -0.25, dtype=judgment_distribution.dtype).to(judgment_distribution.device)

    new_dims = [1] * (judgment_distribution.ndim-1) + [n_class,]
    return (weights.view(*new_dims) * judgment_distribution).sum(-1)


def normalized_score_to_logits(scores):
    probs = torch.stack([1-scores, scores], dim=-1)
    logits = probs.log()
    return probs, logits

class DirichletMultinomialLoss(nn.Module):
    """Negative log-likelihood for a dirichlet-multinomial."""

    # N.B. note that this function computes the likelihood of the
    # observed labels, and not the likelihood of the sufficient
    # statistic derived from them (i.e., the counts of each label). We
    # only need the sufficient statistic however to compute this
    # likelihood, and both lead to the same MLE.
    def forward(self, inputs, targets):  # logits vs freq
        inputs = torch.exp(inputs)
        return - torch.mean(
            torch.lgamma(torch.sum(inputs, dim=-1)) + torch.sum(torch.lgamma(inputs + targets), dim=-1)
            - torch.lgamma(torch.sum(inputs + targets, dim=-1))
            - torch.sum(torch.lgamma(inputs), dim=-1))

def calculate_binary_logits_loss(
        logits, labels,
        problem_type="single_label_classification",
        **kwargs
):
    # assert logits.shape[-1] == 2
    batch_size, nc = logits.shape

    if problem_type == "single_label_classification":
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, nc), labels.view(-1))
    elif problem_type == "soft_label":
        soft_labels = kwargs.get("soft_labels")
        soft_weight = torch.max(soft_labels, dim=-1)[0]
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, nc), labels.view(-1)) * soft_weight
        loss = torch.mean(losses)
    elif problem_type == "full_soft_label":
        soft_labels = kwargs.get("soft_labels")
        norm_fct = nn.LogSoftmax(dim=-1)
        pred_logprob = norm_fct(logits.view(-1, nc))
        loss = - torch.mean(torch.sum(pred_logprob * soft_labels, dim=-1))
    elif problem_type == "thresh_soft_label":
        soft_labels = kwargs.get("soft_labels")
        soft_mask1 = soft_labels >= 0.333
        soft_mask2 = torch.arange(0, nc, device=labels.device, dtype=labels.dtype).unsqueeze(0) == \
                     labels.unsqueeze(1)
        soft_mask = torch.logical_or(soft_mask1, soft_mask2)
        soft_labels = soft_mask.to(soft_labels.dtype) * soft_labels
        norm_fct = nn.LogSoftmax(dim=-1)
        pred_logprob = norm_fct(logits.view(-1, nc))
        loss = - torch.mean(torch.sum(pred_logprob * soft_labels, dim=-1))
    elif problem_type == "soft_label_weighted":
        weights = kwargs.get("weights")
        soft_labels = kwargs.get("soft_labels")
        soft_weight = torch.max(soft_labels, dim=-1)[0]
        loss_fct = nn.CrossEntropyLoss(weight=weights, reduction="none")
        losses = loss_fct(logits.view(-1, nc), labels.view(-1)) * soft_weight
        loss = torch.mean(losses)
    elif problem_type == "full_soft_label_weighted":
        weights = kwargs.get("weights")  # [nc]

        soft_labels = kwargs.get("soft_labels")
        soft_mask1 = soft_labels > 0.
        soft_mask2 = torch.arange(0, nc, device=labels.device, dtype=labels.dtype).unsqueeze(0) == \
                     labels.unsqueeze(1)
        soft_mask = torch.logical_or(soft_mask1, soft_mask2)

        # add_weight to mask
        soft_labels = soft_mask.to(soft_labels.dtype) * soft_labels  # [..., nc]
        soft_labels = soft_labels * weights

        norm_fct = nn.LogSoftmax(dim=-1)
        pred_logprob = norm_fct(logits.view(-1, nc))
        loss = - torch.mean(torch.sum(pred_logprob * soft_labels, dim=-1))

    elif problem_type == "thresh_soft_label_weighted":
        weights = kwargs.get("weights")  # [nc]

        soft_labels = kwargs.get("soft_labels")
        soft_mask1 = soft_labels >= 0.333
        soft_mask2 = torch.arange(0, nc, device=labels.device, dtype=labels.dtype).unsqueeze(0) == \
                     labels.unsqueeze(1)
        soft_mask = torch.logical_or(soft_mask1, soft_mask2)

        # add_weight to mask
        soft_labels = soft_mask.to(soft_labels.dtype) * soft_labels  # [..., nc]
        soft_labels = soft_labels * weights

        norm_fct = nn.LogSoftmax(dim=-1)
        pred_logprob = norm_fct(logits.view(-1, nc))
        loss = - torch.mean(torch.sum(pred_logprob * soft_labels, dim=-1))
    elif problem_type == "rank":
        pos_scores = logits[range(batch_size), labels]
        neg_scores = logits[range(batch_size), 1-labels]
        # losses = torch.relu(0.25 - pos_scores + neg_scores)
        losses = torch.relu(kwargs.get("margin", 0.25) - pos_scores + neg_scores)
        loss = losses.mean()
    elif problem_type == "dirichlet":
        loss_fct = DirichletMultinomialLoss()
        freq_labels = kwargs.get("freq_labels")
        loss = loss_fct(inputs=logits, targets=freq_labels)
    else:
        raise NotImplementedError

    return loss


def xentropy(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> np.float64:
    """Return the xentropy of ``y_pred`` with respect to ``y_true``.
    Parameters
    ----------
    y_true : np.ndarray, required
        An ``n_samples`` by ``n_classes`` array for the class
        probabilities given to each sample.
    y_pred : np.ndarray, required
        An ``n_samples`` by ``n_classes`` array for the predicted class
        probabilities given to each sample.
    Returns
    -------
    np.float64
        The xentropy of ``y_pred` with respect to ``y_true``.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(- np.sum(np.log(y_pred ** y_true), axis=1))


def calibration_factor(
        logits: np.ndarray,
        targets: np.ndarray
) -> np.float64:
    """Return the calibrating temperature for the model.
    Parameters
    ----------
    logits : np.ndarray, required
        The logits from the model to calibrate.
    targets : np.ndarray, required
        The targets on which to calibrate. The targets should be probabilities.
    Returns
    -------
    np.float64
        The temperature to use when calibrating the model. Divide the logits by
        this number to calibrate them.
    """
    def loss(t):
        return xentropy(y_true=targets, y_pred=softmax(logits / t, axis=-1))

    return minimize_scalar(
        fun=loss,
        bounds=(1e-10, 1e10),
    ).x


def calibration_factor_for_mixed_distribution(
    logits_list,
    targets,
):
    assert len(logits_list) == 3

    def loss(args):
        t0, t1, t2 = args
        # dist_x = (np.expand_dims(dist0, -2) * np.expand_dims(dist0, -1)).reshape(-1, 4)
        # dist_x = dist_x.reshape(-1, 4)
        dist0, dist1 = softmax(logits_list[0] / t0, axis=-1), softmax(logits_list[1] / t1, axis=-1)
        dist_if = softmax(logits_list[2] / t2, axis=-1)
        all_dist = np.stack(
            [
                dist_if[:, 0] * dist0[:, 0] * dist1[:, 0],
                dist_if[:, 0] * dist0[:, 1] * dist1[:, 0],
                dist_if[:, 0] * dist0[:, 0] * dist1[:, 1],
                dist_if[:, 0] * dist0[:, 1] * dist1[:, 1],
                dist_if[:, 1],
            ], axis=-1
        )
        return xentropy(y_true=targets, y_pred=all_dist)

    res = minimize(
        loss,
        x0=np.ones([3], dtype="float64"),
        bounds=[(1e-10, 1e10),(1e-10, 1e10),(1e-10, 1e10)],
    )
    logging.info(res)
    return res.x


def calibration_factor_for_mixed_distribution_ONE(
    logits_list,
    targets,
):
    assert len(logits_list) == 3
    dist0, dist1 = softmax(logits_list[0], axis=-1), softmax(logits_list[1], axis=-1)
    dist_if = softmax(logits_list[2], axis=-1)
    all_dist = np.stack(
        [
            dist_if[:, 0] * dist0[:, 0] * dist1[:, 0],
            dist_if[:, 0] * dist0[:, 1] * dist1[:, 0],
            dist_if[:, 0] * dist0[:, 0] * dist1[:, 1],
            dist_if[:, 0] * dist0[:, 1] * dist1[:, 1],
            dist_if[:, 1],
        ], axis=-1
    )

    def loss(t):
        return xentropy(y_true=targets, y_pred=softmax(np.log(all_dist) / t, axis=-1))

    return minimize_scalar(
        fun=loss,
        bounds=(1e-10, 1e10),
    ).x



