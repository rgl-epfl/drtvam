import drjit as dr
import mitsuba as mi


def relu(x):
    return dr.select(x > 0, x, 0)


class Loss:
    def __init__(self, props):
        reduction = props.get('reduction', 'sum')
        if reduction == 'sum':
            self.reduction = dr.sum
        elif reduction == 'mean':
            self.reduction = dr.mean
        else:
            raise ValueError(f"Invalid reduction method: '{reduction}'.")

    def eval_in(self, x):
        raise NotImplementedError

    def eval_out(self, x):
        raise NotImplementedError

    def eval(self, x, target, patterns):
        raise NotImplementedError

    def __call__(self, x, target, patterns):
        if x.shape != target.shape:
            if len(x.shape) == len(target.shape) + 1 and x.shape[-1] == 1:
                # we expect the last dimension to have 1 or 2 channels
                target = target[..., None]
            else:
                raise ValueError(f"Input and target shapes do not match: \
                                 {x.shape} != {target.shape}")

        if target.shape[-1] == 1:  # binary or grayscale target
            loss, loss_patterns = self.eval(x, target, patterns)
        elif target.shape[-1] == 2:  # Surface-aware discretization
            # Here, the target defines the fractional inside/outside
            # volumes of individual voxels.
            w_in = target[..., 0] / (target[..., 0] + target[..., 1])
            w_out = target[..., 1] / (target[..., 0] + target[..., 1])

            loss = w_in * self.eval_in(x[..., 0]) +\
                w_out * self.eval_out(x[..., 1])
            loss_patterns = self.eval_sparsity(patterns)
        else:
            raise ValueError(f"[Loss] Received tensors of invalid shape: \
                             {target.shape}. The last dimension should be\
                             either 1 or 2.")

        mi.Log(mi.LogLevel.Debug, "loss_patterns {}".format(\
               self.reduction(loss_patterns, axis=None)))

        # loss_patterns and loss are still arrays but with different shapes.
        # Hence separate reduction
        return self.reduction(loss, axis=None) +\
            self.reduction(loss_patterns, axis=None)


# TODO: implement L1 in an example in the documentation
class L2Loss(Loss):
    def __init__(self, props):
        super().__init__(props)
        self.M = props.get('M', 4)
        # by default no sparsity
        self.weight_sparsity = props.get('weight_sparsity', 0)

    def eval_in(self, x):
        return dr.square(x - 1.)

    def eval_out(self, x):
        return dr.square(x)

    def eval(self, x, target, patterns):
        return dr.square(x - target), 0 * patterns

    def eval_sparsity(self, patterns):
        return patterns**self.M * self.weight_sparsity

class ThresholdedLoss(Loss):
    """
    Thresholded loss following Wechsler et al 2024.

    The loss is defined as:
    L(x, patterns) = weight_object * relu(tu - x)^K +
                     weight_void * relu(x - tl)^K +
                     weight_limit * relu(x - t_limit)^K +
                     patterns * weight_sparsity

    where:
    - x is the intensity distribution in the printing region
    - tu is the upper threshold, by default 0.95
    - tl is the lower threshold, by default 0.9
    - t_limit is the limit for the overpolymerization term, by default 1.0
    - weight_object and weight_void are the weights for the object/void regions
    - weight_limit is the limit for the overpolymerization term
    - K is the exponent for the loss function, by default 2
    - M is the exponent for the sparsity term, by default 4
    - weight_sparsity is the weight for the sparsity term, by default 0
    """
    def __init__(self, props):
        super().__init__(props)
        self.K = props.get('K', 2)
        self.M = props.get('M', 4)
        self.tl = props.get('tl', 0.9)
        self.tu = props.get('tu', 0.95)
        self.t_limit = props.get('t_limit', 1.0)
        # put a different weight for the object and void regions
        self.weight_object = props.get('weight_object', 1)
        self.weight_void = props.get('weight_void', 1)
        self.weight_limit = props.get('weight_limit', 1)
        # by default no sparsity
        self.weight_sparsity = props.get('weight_sparsity', 0)

        if self.tl >= self.tu:
            raise ValueError(f"[ThresholdedLoss] Lower threshold ({self.tl})\
                             must be smaller than upper threshold ({self.tu})")

    def eval_in(self, x):
        return self.weight_object * relu(self.tu - x)**self.K +\
            self.weight_limit * relu(x - self.t_limit)**self.K

    def eval_out(self, x):
        return self.weight_void * relu(x - self.tl)**self.K

    def eval_sparsity(self, patterns):
        return dr.abs(patterns)**self.M * self.weight_sparsity

    def eval(self, x, target, patterns):
        # target should be a binary inside/outside mask
        return dr.select(target > 0, self.eval_in(x), self.eval_out(x)),\
                self.eval_sparsity(patterns)



losses = {
    'l2': L2Loss,
    'threshold': ThresholdedLoss
}
