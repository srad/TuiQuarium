//! Reduced equilibrium model for calibrating stable lake regimes.

use crate::stats::DailyEcologySample;

const DIM: usize = 7;
const MIN_STATE: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducedState {
    pub m: f64,
    pub a: f64,
    pub d: f64,
    pub g: f64,
    pub b: f64,
    pub n: f64,
    pub p: f64,
}

/// Observable coarse metrics that can be compared directly to archived
/// simulation history without inventing hidden guild splits or a biomass
/// mapping for `phytoplankton_load`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ObservableLakeMetrics {
    pub consumer_to_macrophyte_ratio: f64,
    pub detritus_to_macrophyte_ratio: f64,
    pub dissolved_n: f64,
    pub dissolved_p: f64,
    pub phytoplankton_load: f64,
}

impl ObservableLakeMetrics {
    pub fn mean(samples: &[Self]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut mean = Self::default();
        for sample in samples {
            mean.consumer_to_macrophyte_ratio += sample.consumer_to_macrophyte_ratio;
            mean.detritus_to_macrophyte_ratio += sample.detritus_to_macrophyte_ratio;
            mean.dissolved_n += sample.dissolved_n;
            mean.dissolved_p += sample.dissolved_p;
            mean.phytoplankton_load += sample.phytoplankton_load;
        }

        let denom = samples.len() as f64;
        mean.consumer_to_macrophyte_ratio /= denom;
        mean.detritus_to_macrophyte_ratio /= denom;
        mean.dissolved_n /= denom;
        mean.dissolved_p /= denom;
        mean.phytoplankton_load /= denom;
        Some(mean)
    }
}

impl ReducedState {
    pub const fn new(m: f64, a: f64, d: f64, g: f64, b: f64, n: f64, p: f64) -> Self {
        Self {
            m,
            a,
            d,
            g,
            b,
            n,
            p,
        }
    }

    pub fn as_array(self) -> [f64; DIM] {
        [self.m, self.a, self.d, self.g, self.b, self.n, self.p]
    }

    pub fn from_array(values: [f64; DIM]) -> Self {
        Self {
            m: values[0],
            a: values[1],
            d: values[2],
            g: values[3],
            b: values[4],
            n: values[5],
            p: values[6],
        }
    }

    pub fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self::from_array(self.as_array().map(f))
    }

    pub fn max_abs(self) -> f64 {
        self.as_array()
            .into_iter()
            .map(f64::abs)
            .fold(0.0, f64::max)
    }

    pub fn l2_norm(self) -> f64 {
        self.as_array()
            .into_iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
    }

    pub fn relative_log_distance(self, other: Self) -> f64 {
        self.as_array()
            .into_iter()
            .zip(other.as_array())
            .map(|(lhs, rhs)| (lhs.max(MIN_STATE).ln() - rhs.max(MIN_STATE).ln()).abs())
            .fold(0.0, f64::max)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReducedEquilibriumModel {
    pub i0: f64,
    pub z_m: f64,
    pub z_a: f64,
    pub k_w: f64,
    pub k_a: f64,
    pub k_c: f64,
    pub c_m: f64,
    pub k_i_m: f64,
    pub k_n_m: f64,
    pub k_p_m: f64,
    pub rho_n_m: f64,
    pub rho_p_m: f64,
    pub k_i_a: f64,
    pub k_n_a: f64,
    pub k_p_a: f64,
    pub mu_m: f64,
    pub mu_a: f64,
    pub m_m: f64,
    pub m_a: f64,
    pub tau_m0: f64,
    pub tau_ms: f64,
    pub tau_a: f64,
    pub c_a: f64,
    pub g_mg: f64,
    pub g_ag: f64,
    pub alpha_m: f64,
    pub alpha_a: f64,
    pub alpha_g: f64,
    pub alpha_b: f64,
    pub r_d: f64,
    pub g_dg: f64,
    pub g_db: f64,
    pub eps_mg: f64,
    pub eps_ag: f64,
    pub eps_dg: f64,
    pub eps_db: f64,
    pub m_g: f64,
    pub m_b: f64,
    pub q_g: f64,
    pub q_b: f64,
    pub l_n: f64,
    pub l_p: f64,
    pub rho_sn: f64,
    pub rho_sp: f64,
    pub fix_n_threshold: f64,
    pub fix_n_rate: f64,
    pub eta_n: f64,
    pub eta_p: f64,
    pub u_nm: f64,
    pub u_na: f64,
    pub u_pm: f64,
    pub u_pa: f64,
    pub xi_n: f64,
    pub xi_p: f64,
    pub b_p: f64,
    pub p_target: f64,
    pub sigma_nm: f64,
    pub sigma_pm: f64,
}

impl ReducedEquilibriumModel {
    pub fn reference_clearwater() -> (Self, ReducedState) {
        let target = ReducedState::new(6.0, 1.4, 0.8, 0.45, 0.24, 5.2, 0.9);
        let mut model = Self {
            i0: 1.0,
            z_m: 0.65,
            z_a: 0.18,
            k_w: 0.45,
            k_a: 0.55,
            k_c: 0.12,
            c_m: 0.35,
            k_i_m: 0.12,
            k_n_m: 2.5,
            k_p_m: 0.45,
            rho_n_m: 0.05,
            rho_p_m: 0.03,
            k_i_a: 0.18,
            k_n_a: 1.8,
            k_p_a: 0.30,
            mu_m: 0.0,
            mu_a: 0.0,
            m_m: 0.10,
            m_a: 0.12,
            tau_m0: 0.04,
            tau_ms: 0.08,
            tau_a: 0.14,
            c_a: 0.08,
            g_mg: 0.015,
            g_ag: 0.060,
            alpha_m: 0.10,
            alpha_a: 0.0,
            alpha_g: 0.20,
            alpha_b: 0.25,
            r_d: 0.10,
            g_dg: 0.35,
            g_db: 0.50,
            eps_mg: 0.20,
            eps_ag: 0.45,
            eps_dg: 0.0,
            eps_db: 0.0,
            m_g: 0.16,
            m_b: 0.14,
            q_g: 0.10,
            q_b: 0.08,
            l_n: 0.0,
            l_p: 0.0,
            rho_sn: 0.08,
            rho_sp: 0.05,
            fix_n_threshold: 10.0,
            fix_n_rate: 0.24,
            eta_n: 0.55,
            eta_p: 0.20,
            u_nm: 0.11,
            u_na: 0.08,
            u_pm: 0.015,
            u_pa: 0.012,
            xi_n: 0.02,
            xi_p: 0.10,
            b_p: 0.20,
            p_target: 1.2,
            sigma_nm: 0.10,
            sigma_pm: 0.05,
        };

        let u_m = model.u_m(target);
        let tau_m = model.tau_m(target);
        model.mu_m = (model.m_m + tau_m + model.g_mg * target.g) / u_m.max(1e-6);

        let u_a = model.u_a(target);
        model.mu_a = (model.m_a + model.tau_a + model.c_a * target.a + model.g_ag * target.g)
            / u_a.max(1e-6);

        model.eps_dg = (model.m_g + model.q_g * (target.g + target.b)
            - model.eps_mg * model.g_mg * target.m
            - model.eps_ag * model.g_ag * target.a)
            / (model.g_dg * target.d).max(1e-6);
        model.eps_db =
            (model.m_b + model.q_b * (target.g + target.b)) / (model.g_db * target.d).max(1e-6);

        let plant_loss = (model.m_m + tau_m + model.g_mg * target.g) * target.m;
        let algae_loss =
            (model.m_a + model.tau_a + model.c_a * target.a + model.g_ag * target.g) * target.a;
        model.alpha_a = (model.r_d * target.d
            + model.g_dg * target.g * target.d
            + model.g_db * target.b * target.d
            - model.alpha_m * plant_loss
            - model.alpha_g * model.m_g * target.g
            - model.alpha_b * model.m_b * target.b)
            / algae_loss.max(1e-6);

        let n_uptake = model.u_nm * model.mu_m * model.u_m(target) * target.m
            + model.u_na * model.mu_a * model.u_a(target) * target.a;
        let p_uptake = model.u_pm * model.mu_m * model.u_m(target) * target.m
            + model.u_pa * model.mu_a * model.u_a(target) * target.a;
        let s_n = model.s_n_star(target);
        let s_p = model.s_p_star(target);
        model.l_n = n_uptake
            - model.rho_sn * s_n
            - model.fix_n(target.n)
            - model.eta_n * model.r_d * target.d
            + model.xi_n * target.n;
        model.l_p = p_uptake - model.rho_sp * s_p - model.eta_p * model.r_d * target.d
            + model.burial_p(target.p)
            + model.xi_p * target.p;

        (model, target)
    }

    pub fn in_default_realism_band(&self, state: ReducedState) -> bool {
        let m_a = state.m / state.a.max(MIN_STATE);
        let g_m = state.g / state.m.max(MIN_STATE);
        let b_m = state.b / state.m.max(MIN_STATE);
        let d_m = state.d / state.m.max(MIN_STATE);
        state.m > 0.0
            && state.a > 0.0
            && state.g > 0.0
            && state.b > 0.0
            && state.d > 0.0
            && (2.0..=8.0).contains(&m_a)
            && (0.03..=0.14).contains(&g_m)
            && (0.02..=0.10).contains(&b_m)
            && (0.05..=0.35).contains(&d_m)
            && (2.0..=12.0).contains(&state.n)
            && (0.3..=2.5).contains(&state.p)
    }

    /// Project one archived daily sample into the subset of reduced-model
    /// observables that are already recorded by the simulation.
    pub fn observable_metrics_from_daily_sample(
        sample: &DailyEcologySample,
    ) -> ObservableLakeMetrics {
        let producer_total = (sample.producer_total_biomass as f64).max(MIN_STATE);
        ObservableLakeMetrics {
            consumer_to_macrophyte_ratio: sample.consumer_biomass as f64 / producer_total,
            detritus_to_macrophyte_ratio: sample.detritus_energy as f64 / producer_total,
            dissolved_n: sample.dissolved_n as f64,
            dissolved_p: sample.dissolved_p as f64,
            phytoplankton_load: sample.phytoplankton_load as f64,
        }
    }

    pub fn mean_observable_metrics(
        &self,
        samples: &[DailyEcologySample],
    ) -> Option<ObservableLakeMetrics> {
        let projected: Vec<_> = samples
            .iter()
            .map(Self::observable_metrics_from_daily_sample)
            .collect();
        ObservableLakeMetrics::mean(&projected)
    }

    pub fn reference_observable_metrics(&self, state: ReducedState) -> ObservableLakeMetrics {
        ObservableLakeMetrics {
            consumer_to_macrophyte_ratio: (state.g + state.b) / state.m.max(MIN_STATE),
            detritus_to_macrophyte_ratio: state.d / state.m.max(MIN_STATE),
            dissolved_n: state.n,
            dissolved_p: state.p,
            phytoplankton_load: 0.0,
        }
    }

    /// Observable clear-water realism band derived from the reduced model.
    ///
    /// This intentionally excludes exact algal biomass and hidden consumer
    /// guild splits because the current archived history stores only
    /// `phytoplankton_load` and total consumer biomass. The dissolved-P ceiling
    /// is looser than the strict reduced-state band because the archived
    /// simulation includes slower benthic-release and event transients that are
    /// not represented in the reduced steady-state pool; operationally we allow
    /// roughly 5x the reduced equilibrium mean before classifying the run as
    /// outside the clear-water basin.
    pub fn observable_metrics_in_default_realism_band(
        &self,
        metrics: ObservableLakeMetrics,
    ) -> bool {
        (0.01..=0.30).contains(&metrics.consumer_to_macrophyte_ratio)
            && (0.02..=0.40).contains(&metrics.detritus_to_macrophyte_ratio)
            && (2.0..=12.0).contains(&metrics.dissolved_n)
            && (0.3..=4.5).contains(&metrics.dissolved_p)
            && (0.0..=0.35).contains(&metrics.phytoplankton_load)
    }

    pub fn solve_from_guess(
        &self,
        initial_guess: ReducedState,
        max_iters: usize,
    ) -> Option<ReducedState> {
        let mut z = initial_guess.map(|v| v.max(MIN_STATE).ln()).as_array();
        let mut norm = self.residual_norm(ReducedState::from_array(z.map(f64::exp)));

        for _ in 0..max_iters {
            if norm < 1e-10 {
                return Some(ReducedState::from_array(z.map(f64::exp)));
            }
            let jacobian = self.jacobian_in_log_space(z);
            let rhs = self
                .residuals(ReducedState::from_array(z.map(f64::exp)))
                .as_array()
                .map(|v| -v);
            let step = solve_linear_system(jacobian, rhs)?;

            let mut accepted = false;
            let mut alpha = 1.0;
            while alpha >= 1.0 / 1024.0 {
                let trial = add_arrays(z, scale_array(step, alpha));
                let trial_state = ReducedState::from_array(trial.map(f64::exp));
                let trial_norm = self.residual_norm(trial_state);
                if trial_norm < norm {
                    z = trial;
                    norm = trial_norm;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                return None;
            }
        }

        let state = ReducedState::from_array(z.map(f64::exp));
        if self.residual_norm(state) < 1e-8 {
            Some(state)
        } else {
            None
        }
    }

    pub fn step(&self, state: ReducedState, dt: f64) -> ReducedState {
        let delta = self.residuals(state);
        ReducedState::new(
            (state.m + dt * delta.m).max(MIN_STATE),
            (state.a + dt * delta.a).max(MIN_STATE),
            (state.d + dt * delta.d).max(MIN_STATE),
            (state.g + dt * delta.g).max(MIN_STATE),
            (state.b + dt * delta.b).max(MIN_STATE),
            (state.n + dt * delta.n).max(MIN_STATE),
            (state.p + dt * delta.p).max(MIN_STATE),
        )
    }

    pub fn residual_norm(&self, state: ReducedState) -> f64 {
        self.residuals(state).l2_norm()
    }

    pub fn residuals(&self, state: ReducedState) -> ReducedState {
        let (u_m, tau_m, s_n, s_p) = self.macrophyte_terms(state);
        let u_a = self.u_a(state);

        let r_m = self.mu_m * u_m * state.m
            - self.m_m * state.m
            - tau_m * state.m
            - self.g_mg * state.g * state.m;
        let r_a = self.mu_a * u_a * state.a
            - self.m_a * state.a
            - self.tau_a * state.a
            - self.c_a * state.a * state.a
            - self.g_ag * state.g * state.a;
        let plant_loss = self.m_m * state.m + tau_m * state.m + self.g_mg * state.g * state.m;
        let algae_loss = self.m_a * state.a
            + self.tau_a * state.a
            + self.c_a * state.a * state.a
            + self.g_ag * state.g * state.a;
        let r_d = self.alpha_m * plant_loss
            + self.alpha_a * algae_loss
            + self.alpha_g * self.m_g * state.g
            + self.alpha_b * self.m_b * state.b
            - self.r_d * state.d
            - self.g_dg * state.g * state.d
            - self.g_db * state.b * state.d;
        let r_g = self.eps_mg * self.g_mg * state.g * state.m
            + self.eps_ag * self.g_ag * state.g * state.a
            + self.eps_dg * self.g_dg * state.g * state.d
            - self.m_g * state.g
            - self.q_g * state.g * (state.g + state.b);
        let r_b = self.eps_db * self.g_db * state.b * state.d
            - self.m_b * state.b
            - self.q_b * state.b * (state.g + state.b);
        let r_n =
            self.l_n + self.rho_sn * s_n + self.fix_n(state.n) + self.eta_n * self.r_d * state.d
                - self.u_nm * self.mu_m * u_m * state.m
                - self.u_na * self.mu_a * u_a * state.a
                - self.xi_n * state.n;
        let r_p = self.l_p + self.rho_sp * s_p + self.eta_p * self.r_d * state.d
            - self.u_pm * self.mu_m * u_m * state.m
            - self.u_pa * self.mu_a * u_a * state.a
            - self.burial_p(state.p)
            - self.xi_p * state.p;

        ReducedState::new(r_m, r_a, r_d, r_g, r_b, r_n, r_p)
    }

    fn jacobian_in_log_space(&self, z: [f64; DIM]) -> [[f64; DIM]; DIM] {
        let mut jacobian = [[0.0; DIM]; DIM];
        for col in 0..DIM {
            let delta = 1e-6;
            let mut plus = z;
            let mut minus = z;
            plus[col] += delta;
            minus[col] -= delta;
            let r_plus = self
                .residuals(ReducedState::from_array(plus.map(f64::exp)))
                .as_array();
            let r_minus = self
                .residuals(ReducedState::from_array(minus.map(f64::exp)))
                .as_array();
            for row in 0..DIM {
                jacobian[row][col] = (r_plus[row] - r_minus[row]) / (2.0 * delta);
            }
        }
        jacobian
    }

    fn light_m(&self, state: ReducedState) -> f64 {
        let canopy = self.c_m * state.m;
        self.i0 * (-(self.k_w * self.z_m + self.k_a * state.a + self.k_c * canopy)).exp()
    }

    fn light_a(&self, state: ReducedState) -> f64 {
        self.i0 * (-(self.k_w * self.z_a + self.k_a * state.a)).exp()
    }

    fn macrophyte_terms(&self, state: ReducedState) -> (f64, f64, f64, f64) {
        let l_i = sat(self.light_m(state), self.k_i_m);
        let mut tau_m = self.tau_m0;
        let mut s_n = ((1.0 - self.eta_n) * self.r_d * state.d + self.sigma_nm * tau_m * state.m)
            / self.rho_sn.max(1e-6);
        let mut s_p = ((1.0 - self.eta_p) * self.r_d * state.d
            + self.sigma_pm * tau_m * state.m
            + self.burial_p(state.p))
            / self.rho_sp.max(1e-6);

        for _ in 0..24 {
            let effective_n = state.n + self.rho_n_m * s_n;
            let effective_p = state.p + self.rho_p_m * s_p;
            let u_m = l_i
                .min(sat(effective_n, self.k_n_m))
                .min(sat(effective_p, self.k_p_m));
            let next_tau_m = self.tau_m0 + self.tau_ms * (1.0 - u_m).powi(2);
            let next_s_n = ((1.0 - self.eta_n) * self.r_d * state.d
                + self.sigma_nm * next_tau_m * state.m)
                / self.rho_sn.max(1e-6);
            let next_s_p = ((1.0 - self.eta_p) * self.r_d * state.d
                + self.sigma_pm * next_tau_m * state.m
                + self.burial_p(state.p))
                / self.rho_sp.max(1e-6);

            if (next_tau_m - tau_m).abs() < 1e-12
                && (next_s_n - s_n).abs() < 1e-12
                && (next_s_p - s_p).abs() < 1e-12
            {
                tau_m = next_tau_m;
                s_n = next_s_n;
                s_p = next_s_p;
                break;
            }

            tau_m = next_tau_m;
            s_n = next_s_n;
            s_p = next_s_p;
        }

        let effective_n = state.n + self.rho_n_m * s_n;
        let effective_p = state.p + self.rho_p_m * s_p;
        let u_m = l_i
            .min(sat(effective_n, self.k_n_m))
            .min(sat(effective_p, self.k_p_m));

        (u_m, tau_m, s_n, s_p)
    }

    fn u_m(&self, state: ReducedState) -> f64 {
        self.macrophyte_terms(state).0
    }

    fn u_a(&self, state: ReducedState) -> f64 {
        let l_i = sat(self.light_a(state), self.k_i_a);
        let l_n = sat(state.n, self.k_n_a);
        let l_p = sat(state.p, self.k_p_a);
        l_i.min(l_n).min(l_p)
    }

    fn tau_m(&self, state: ReducedState) -> f64 {
        self.macrophyte_terms(state).1
    }

    fn s_n_star(&self, state: ReducedState) -> f64 {
        self.macrophyte_terms(state).2
    }

    fn s_p_star(&self, state: ReducedState) -> f64 {
        self.macrophyte_terms(state).3
    }

    fn burial_p(&self, p: f64) -> f64 {
        self.b_p * (p - self.p_target).max(0.0)
    }

    fn fix_n(&self, n: f64) -> f64 {
        if n < self.fix_n_threshold {
            self.fix_n_rate * (1.0 - n / self.fix_n_threshold)
        } else {
            0.0
        }
    }
}

fn sat(resource: f64, half_sat: f64) -> f64 {
    if resource <= 0.0 {
        0.0
    } else {
        resource / (resource + half_sat.max(1e-9))
    }
}

fn add_arrays(lhs: [f64; DIM], rhs: [f64; DIM]) -> [f64; DIM] {
    let mut out = [0.0; DIM];
    for i in 0..DIM {
        out[i] = lhs[i] + rhs[i];
    }
    out
}

fn scale_array(values: [f64; DIM], factor: f64) -> [f64; DIM] {
    let mut out = [0.0; DIM];
    for i in 0..DIM {
        out[i] = values[i] * factor;
    }
    out
}

fn solve_linear_system(mut a: [[f64; DIM]; DIM], mut b: [f64; DIM]) -> Option<[f64; DIM]> {
    for pivot in 0..DIM {
        let mut max_row = pivot;
        let mut max_value = a[pivot][pivot].abs();
        for row in (pivot + 1)..DIM {
            let value = a[row][pivot].abs();
            if value > max_value {
                max_value = value;
                max_row = row;
            }
        }
        if max_value < 1e-12 {
            return None;
        }
        if max_row != pivot {
            a.swap(pivot, max_row);
            b.swap(pivot, max_row);
        }

        let pivot_value = a[pivot][pivot];
        for col in pivot..DIM {
            a[pivot][col] /= pivot_value;
        }
        b[pivot] /= pivot_value;

        for row in 0..DIM {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            if factor.abs() < 1e-12 {
                continue;
            }
            for col in pivot..DIM {
                a[row][col] -= factor * a[pivot][col];
            }
            b[row] -= factor * b[pivot];
        }
    }

    Some(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_clearwater_state_satisfies_equilibrium() {
        let (model, target) = ReducedEquilibriumModel::reference_clearwater();
        let residual = model.residuals(target);
        assert!(
            model.in_default_realism_band(target),
            "Reference state should sit inside the default realism band: {target:?}"
        );
        assert!(
            residual.max_abs() < 1e-9,
            "Reference clear-water state should satisfy the reduced equilibrium. residual={residual:?}"
        );
    }

    #[test]
    fn solver_recovers_reference_clearwater_solution() {
        let (model, target) = ReducedEquilibriumModel::reference_clearwater();
        let guess = ReducedState::new(
            target.m * 1.25,
            target.a * 0.80,
            target.d * 1.30,
            target.g * 0.78,
            target.b * 1.18,
            target.n * 0.82,
            target.p * 1.22,
        );

        let solved = model
            .solve_from_guess(guess, 24)
            .expect("Newton solve should converge from a nearby positive guess");

        assert!(
            solved.relative_log_distance(target) < 1e-6,
            "Solved state should recover the reference equilibrium. solved={solved:?} target={target:?}"
        );
        assert!(
            model.residual_norm(solved) < 1e-8,
            "Solved state should have a near-zero residual, got {:.3e}",
            model.residual_norm(solved),
        );
    }

    #[test]
    fn local_perturbations_relax_toward_reference_state() {
        let (model, target) = ReducedEquilibriumModel::reference_clearwater();
        let factors = [0.95, 1.05];
        let mut worst_final_distance: f64 = 0.0;

        for index in 0..DIM {
            for factor in factors {
                let mut perturbed = target.as_array();
                perturbed[index] *= factor;
                let initial = ReducedState::from_array(perturbed);
                let initial_distance = initial.relative_log_distance(target);
                let mut state = initial;
                for _ in 0..40 {
                    state = model.step(state, 0.15);
                }
                let final_distance = state.relative_log_distance(target);
                worst_final_distance = worst_final_distance.max(final_distance);
                assert!(
                    final_distance < initial_distance,
                    "Perturbation in dimension {index} with factor {factor} should relax toward equilibrium. initial={initial:?} final={state:?}"
                );
            }
        }

        assert!(
            worst_final_distance < 0.12,
            "Perturbation relaxation should stay near the reference regime, worst final log-distance={worst_final_distance:.4}"
        );
    }

    #[test]
    fn reference_observable_metrics_match_clearwater_band() {
        let (model, target) = ReducedEquilibriumModel::reference_clearwater();
        let metrics = model.reference_observable_metrics(target);

        assert!(
            model.observable_metrics_in_default_realism_band(metrics),
            "Reference observable metrics should sit inside the clear-water realism band: {metrics:?}"
        );
    }
}
