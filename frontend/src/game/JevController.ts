import type { JevDecision } from './types';

/** Physics consumes a cached action, independently of the selected policy. */
export interface PolicyController {
  decide(): JevDecision;
}

// Compatibility for existing integrations.
export type JevController = PolicyController;
