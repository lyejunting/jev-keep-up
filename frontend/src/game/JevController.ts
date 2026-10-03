import type { JevDecision } from './types';

// Read the latest decision without exposing physics coordinates to JEV.
export interface JevController {
  decide(): JevDecision;
}
