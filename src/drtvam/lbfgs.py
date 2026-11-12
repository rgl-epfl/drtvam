import mitsuba as mi
import drjit as dr
from collections import defaultdict

class LBFGS(mi.ad.Optimizer):
    def __init__(self, lr=1.0, m=5, params=None, line_search_fn=None, wolfe=False, search_it=20, init_alpha=None):
        super().__init__(lr, params)
        # Use Wolfe conditions, otherwise only use Armijo condition
        self.wolfe = wolfe
        # History size
        self.m = m
        # Function used for line search, should take a dictionary of variables
        # as input and return the loss and gradients for each variable
        self.line_search_fn = line_search_fn
        # How many iterations of backtracking line search to run
        self.search_it = search_it
        # Initial step size of the line search.
        if init_alpha is None:
            init_alpha = 10. if wolfe else 1.
        self.init_alpha = init_alpha

    def _reset(self, key, value, promoted):
        self.state[key] = value, promoted, None, ([], [], [], 0, None, None)

    def update_history(self, k):
        # Discard too old entries
        val, promoted, lr, (s, y, ys, t, p_old, g_old) = self.state[k]

        if t > self.m:
            s.pop(0)
            y.pop(0)
            ys.pop(0)

        p = dr.ravel(dr.detach(val))
        g_p = dr.ravel(val.grad)
        dr.schedule(p, g_p)
        # Update history
        if t > 0:
            s.append(p - p_old)
            y.append(g_p - g_old)
            dr.schedule(s[-1], y[-1])
            ys.append(dr.dot(y[-1], s[-1]))

        # Update previous state and gradient
        self.state[k] = val, promoted, lr, (s, y, ys, t+1, p, g_p)

    def step(self, f):
        search_dirs = {}
        # TODO: unravel arrays for dot products

        # Compute search directions
        for k in self.state.keys():
            # Update history
            self.update_history(k)
            p, promoted, lr, (s, y, ys, t, p_old, g_old) = self.state[k]

            # Stopping optimization early because parameter or gradient change is too small
            if t > 1 and dr.all(ys[-1] == 0):
               return False

            # Find search direction
            q = dr.ravel(p.grad)
            hist_size = len(s)
            dr.make_opaque(hist_size)
            alphas = []
            for i in range(hist_size-1, -1, -1):
                rho = dr.rcp(ys[i])
                a = rho * dr.dot(s[i], q)
                q = q - a * y[i]
                alphas.insert(0, a)

            gamma = 1 if t == 1 else ys[-1] / dr.dot(y[-1], y[-1])

            z = gamma * q
            for i in range(hist_size):
                rho = dr.rcp(ys[i])
                b = rho * dr.dot(y[i], z)
                z = z + (alphas[i] - b) * s[i]

            if type(p) == mi.TensorXf:
                search_dirs[k] = dr.detach(type(p)(-z, shape=p.shape))
            else:
                search_dirs[k] = dr.detach(dr.unravel(type(p), -z))

        # Backtracking line search until Wolfe conditions are satisfied /!\ The
        # final step size is the same for all variables, not sure if this is the
        # proper way to do it
        alpha = dr.opaque(mi.Float, self.init_alpha)
        c1 = 1e-4
        c2 = 0.9
        for it in range(self.search_it):
            params = {}
            for k, (p, _, _, _) in self.state.items():
                params[k] = dr.detach(p + alpha * search_dirs[k])

                if self.wolfe:
                    dr.enable_grad(params[k])

                dr.schedule(params[k])

            f_new = self.line_search_fn(params)
            if self.wolfe:
                dr.backward(f_new)

            g_dot_z = mi.Float(0.)
            g_new_dot_z = mi.Float(0.)
            for k in self.state.keys():
                g_old = self.state[k][3][5]
                g_dot_z += dr.dot(g_old, dr.ravel(search_dirs[k]))
                if self.wolfe:
                    g_new_dot_z += dr.dot(dr.ravel(params[k].grad), dr.ravel(search_dirs[k]))

            wolfe1 = (f_new <= f + c1 * alpha * g_dot_z)
            if not self.wolfe:
                if dr.all(wolfe1):
                    break
            else:
                wolfe2 = (g_new_dot_z >= c2 * g_dot_z)
                if dr.all(wolfe1 & wolfe2):
                    break

            alpha *= 0.5

        for k, (p, promoted, lr, extra) in self.state.items():
            new_val = dr.detach(p + alpha * search_dirs[k])

            dr.enable_grad(new_val)
            new_state = new_val, promoted, lr, extra
            dr.schedule(new_state)
            self.state[k] = new_state

        return True

class LinearLBFGS(mi.ad.Optimizer):
    def __init__(self, lr=1.0, m=5, params=None, render_fn=None, loss_fn=None, search_it=20, init_alpha=1.):
        super().__init__(lr, params)
        # History size
        self.m = m
        # Function used for line search, should take a dictionary of variables
        # as input and return the loss and gradients for each variable
        self.render_fn = render_fn
        self.loss_fn = loss_fn
        # How many iterations of backtracking line search to run
        self.search_it = search_it
        # Initial step size of the line search.
        self.init_alpha = init_alpha

    def _reset(self, key, value, promoted):
        self.state[key] = value, promoted, None, ([], [], [], 0, None, None)

    def update_history(self, k):
        # Discard too old entries
        val, promoted, lr, (s, y, ys, t, p_old, g_old) = self.state[k]

        if t > self.m:
            s.pop(0)
            y.pop(0)
            ys.pop(0)

        p = dr.ravel(dr.detach(val))
        g_p = dr.ravel(val.grad)
        dr.schedule(p, g_p)
        # Update history
        if t > 0:
            s.append(p - p_old)
            y.append(g_p - g_old)
            dr.schedule(s[-1], y[-1])
            ys.append(dr.dot(y[-1], s[-1]))

        # Update previous state and gradient
        self.state[k] = val, promoted, lr, (s, y, ys, t+1, p, g_p)

    def step(self, vol, loss, fwd):
        search_dirs = {}
        # TODO: unravel arrays for dot products

        # Compute search directions
        for k in self.state.keys():
            # Update history
            self.update_history(k)

            p, promoted, lr, (s, y, ys, t, p_old, g_old) = self.state[k]

            # Stopping optimization early because parameter or gradient change is too small
            if t > 1 and dr.all(ys[-1] == 0):
                return False

            # Find search direction
            q = dr.ravel(p.grad)
            hist_size = len(s)
            dr.make_opaque(hist_size)
            alphas = []
            for i in range(hist_size-1, -1, -1):
                rho = dr.rcp(ys[i])
                a = rho * dr.dot(s[i], q)
                q = q - a * y[i]
                alphas.insert(0, a)

            gamma = 1 if t == 1 else ys[-1] / dr.dot(y[-1], y[-1])

            z = gamma * q
            for i in range(hist_size):
                rho = dr.rcp(ys[i])
                b = rho * dr.dot(y[i], z)
                z = z + (alphas[i] - b) * s[i]

            if type(p) == mi.TensorXf:
                search_dirs[k] = dr.detach(type(p)(-z, shape=p.shape))
            else:
                search_dirs[k] = dr.detach(dr.unravel(type(p), -z))
            #TODO: clamping ?
            dr.schedule(search_dirs[k])

        # Backtracking line search until Armijo conditions are satisfied
        alpha = dr.opaque(mi.Float, self.init_alpha)
        c1 = 1e-4

        dvol = self.render_fn(search_dirs)
        dr.eval(dvol)

        # Compute dot product of gradient and search direction for Armijo condition
        g_dot_z = mi.Float(0.)
        for k in self.state.keys():
            g_old = self.state[k][3][5]
            g_dot_z += dr.dot(g_old, dr.ravel(search_dirs[k]))

        for _ in range(self.search_it):

            out = fwd(vol + alpha * dvol)
            new_patterns = (
                dr.detach(self.state['projector.active_data'][0]) +
                alpha * search_dirs['projector.active_data']
            )
            mi.Log(mi.LogLevel.Debug, "[drtvam] Calling loss from LBFGS")
            f_new = self.loss_fn(out[0], new_patterns, *out[1:])

            armijo = (f_new <= loss + c1 * alpha * g_dot_z)
            if dr.all(armijo):
                break

            alpha *= 0.5

        for k, (p, promoted, lr, extra) in self.state.items():
            new_val = dr.detach(p + alpha * search_dirs[k])

            dr.enable_grad(new_val)
            new_state = new_val, promoted, lr, extra
            dr.schedule(new_state)
            self.state[k] = new_state

        return True
