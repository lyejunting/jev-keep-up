import { POLICY_OPTIONS, predictState } from './services/jevClient';
import { predictTinyMlp } from './services/tinyMlp';
import type { PolicyId } from './services/jevClient';

export const browserOnly = __KEEPUP_BROWSER_ONLY__;
export const availablePolicies = browserOnly ? POLICY_OPTIONS.filter(option => option.id === 'tiny_mlp') : POLICY_OPTIONS;
export const initialModel: PolicyId = browserOnly ? 'tiny_mlp' : 'jev';

export const gamePredictor: typeof predictState = async (state, url, signal, model = initialModel) => {
  if (signal.aborted) throw new Error('Prediction cancelled');
  if (model === 'tiny_mlp') return predictTinyMlp(state);
  if (browserOnly) throw new Error('Jev is only available locally');
  return predictState(state, url, signal, model);
};
