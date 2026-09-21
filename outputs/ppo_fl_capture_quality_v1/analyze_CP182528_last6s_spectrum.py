"""Single sealed-window spectral diagnostic; no resampling or model calls."""
import json
from pathlib import Path
import numpy as np
from analyze_CP182528_P05 import SOURCE, OUT, rows

LOW, HIGH, FS = 5317, 6036, 120.


def spectrum(x, fs):
    x = np.asarray(x, dtype=np.float64)
    w = np.hanning(len(x)); centered = x-x.mean()
    f = np.fft.rfftfreq(len(x), 1/fs)
    z = np.fft.rfft(centered*w)
    psd = abs(z)**2/(fs*np.sum(w*w))
    psd[1:-1 if len(x)%2 == 0 else None] *= 2
    out = dict(samples=len(x), fs_hz=fs, duration_n_over_fs_s=len(x)/fs,
        resolution_hz=fs/len(x), theoretical_nyquist_hz=fs/2,
        mean=float(x.mean()), peak_to_peak=float(np.ptp(x)), demeaned_rms=float(np.sqrt(np.mean(centered**2))))
    if np.ptp(x) < 1e-10:
        return dict(out, constant=True, dominant_frequency_hz=None, dominant_tone_amplitude=None)
    peak = int(np.argmax(psd[1:])+1)
    band = (abs(f-f[peak]) <= fs/len(x)*1.01) & (f > 0)
    out.update(constant=False, dominant_frequency_hz=float(f[peak]),
        dominant_tone_amplitude=float(2*abs(z[peak])/np.sum(w)),
        peak_three_bin_power_fraction=float(psd[band].sum()/psd[1:].sum()),
        power_fraction_above_7p5Hz=float(psd[f>7.5].sum()/psd[1:].sum()))
    return out


def local_phase(x, y, fs, hz):
    # 1-second, 50%-overlapping Hann windows: phase of measured y vs target x.
    n = int(fs); w = np.hanning(n); k = int(round(hz*n/fs))
    X=[]; Y=[]
    for start in range(0, len(x)-n+1, n//2):
        xx=np.asarray(x[start:start+n]); yy=np.asarray(y[start:start+n])
        X.append(np.fft.rfft((xx-xx.mean())*w)[k]); Y.append(np.fft.rfft((yy-yy.mean())*w)[k])
    X=np.asarray(X); Y=np.asarray(Y); cross=Y*X.conj()
    mean_cross=cross.mean(); phase=float(np.angle(mean_cross))
    return dict(frequency_hz=hz, local_windows=len(X), window_s=1., overlap_fraction=.5,
        phase_y_minus_x_deg=float(np.degrees(phase)),
        apparent_phase_lag_ms=-phase/(2*np.pi*hz)*1000,
        coherence=float(abs(mean_cross)**2/(np.mean(abs(X)**2)*np.mean(abs(Y)**2))),
        gain_y_over_x=float(np.sqrt(np.mean(abs(Y)**2)/np.mean(abs(X)**2))),
        local_phase_min_max_deg=[float(np.degrees(np.angle(cross)).min()),float(np.degrees(np.angle(cross)).max())])


def analyze():
    manifest=json.loads((SOURCE/'semantic_video_source_manifest.json').read_text())
    assert manifest['episode_physics_ticks']==HIGH and manifest['policy_sampling_mode']=='deterministic_conditional_mean'
    physical={p['physics_tick']:p for p in rows(SOURCE/'physical_observations.jsonl') if LOW <= p['physics_tick'] <= HIGH}
    native={n['episode_physics_tick']:n for n in rows(SOURCE/'native_tick_audit.jsonl') if LOW <= n['episode_physics_tick'] <= HIGH}
    assert list(physical)==list(native)==list(range(LOW,HIGH+1))
    ticks=list(physical); t=np.array([physical[i]['simulation_time_s'] for i in ticks])
    assert np.max(abs(np.diff(t)-1/FS)) < 1e-12
    signals=dict(FL_hip_N_deg=[],FL_hip_mapper_compensation_deg=[],FL_hip_filtered_REQUEST_120Hz_deg=[],FL_hip_final_deg=[],FL_hip_actual_deg=[],body_omega_world_x_rad_s=[],body_omega_world_y_rad_s=[],body_omega_world_z_rad_s=[])
    for i in ticks:
        p=physical[i]; n=native[i]
        values=[n['nominal_full12'][0],n['native_audit']['native_drive_target_full12'][0]-n['nominal_full12'][0],n['projected_residual_full12'][0],p['commanded_full12'][0],p['actual_full12'][0],*p['base']['angular_velocity_w_rad_s']]
        for key,value in zip(signals,values,strict=True): signals[key].append(value)
    ds=[d for d in rows(SOURCE/'video_policy_decisions.jsonl') if d['start_tick']>=LOW and d['end_tick']<=HIGH and d['end_tick']-d['start_tick']==8]
    assert len(ds)==89 and all(b['start_tick']-a['start_tick']==8 for a,b in zip(ds,ds[1:])) and all(d['request_phase']=='P05' for d in ds)
    native15=dict(FL_hip_conditional_mean_raw=[d['policy_request']['conditional_mean_full12'][0] for d in ds],
        FL_hip_filtered_REQUEST_decision_end_deg=[d['step_info']['projected_residual_full12'][0] for d in ds])
    result=dict(schema='wlr50_clean.CP182528_fixed_P05_last6s_spectrum.v1',source=str(SOURCE),
        physical_tick_window=[LOW,HIGH],physical_samples=720,preprocessing='demean only; Hann-window one-sided periodogram; no interpolation, no upsampling, no linear detrend',
        tone_amplitude_semantics='2*abs(rFFT)/sum(Hann); sinusoidal amplitude of dominant bin, not total square-wave amplitude',
        signals120Hz={key:spectrum(value,FS) for key,value in signals.items()},
        native15Hz={key:spectrum(value,15.) for key,value in native15.items()},
        native15Hz_time_basis=dict(conditional_mean_start_ticks=[ds[0]['start_tick'],ds[-1]['start_tick']],filtered_REQUEST_end_ticks=[ds[0]['end_tick'],ds[-1]['end_tick']],excluded_last_partial_decision=True,original_uniform_samples=89),
        final_to_actual_at_15Hz=local_phase(signals['FL_hip_final_deg'],signals['FL_hip_actual_deg'],FS,15.),
        final_to_body_omega_x_at_15Hz=local_phase(signals['FL_hip_final_deg'],signals['body_omega_world_x_rad_s'],FS,15.),
        limitations=['Only one fixed six-second P05 window; no whole-episode extrapolation.',
            'Native15Hz policy spectrum cannot identify frequencies above7.5Hz; no interpolated high-frequency policy claim.',
            'REQUEST120Hz is an actual logged execution-layer sequence, not upsampled policy means.',
            'Final command is the current-step target; actual q and body omega are post-step. Apparent phase delay is modulo one period, not a separately identified transport delay.',
            'Closed-loop same-frequency coherence is not isolated causal proof; phase alone cannot establish a new CAPS objective.',
            'Body omega uses measured world-frame components; no inferred angular-velocity derivative.'],
        extra_model_forwards=0,new_optimizer_steps=0,new_simulations=0,production_changed=False)
    return result


def main():
    result=analyze()
    path=OUT/'CP182528_last6s_spectrum.json'
    with path.open('x',encoding='utf-8') as s: json.dump(result,s,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
