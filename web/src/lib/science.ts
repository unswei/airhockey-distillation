export const STEP_MS = 20;
export const MASK_START = 5;
export const MASK_LENGTH = 20;
export const STACK_SIZE = 10;
export type PuckTriple = readonly [number, number, number];

export function visibleAt(step: number): boolean {
  return step < MASK_START || step >= MASK_START + MASK_LENGTH;
}

// Explicitly illustrative geometry. These are not simulator coordinates.
export function schematicPosition(step: number, side: number): [number, number] {
  return [0.86 - step * 0.025, side * (step - MASK_START) * 0.01];
}

export function puckInput(step: number, side: number): PuckTriple {
  return visibleAt(step) ? [...schematicPosition(step, side), 1] : [0, 0, 0];
}

export function stackAt(
  step: number,
  side: number,
): { step: number; padding: boolean; value: PuckTriple }[] {
  return Array.from({ length: STACK_SIZE }, (_, i) => {
    const index = step - STACK_SIZE + 1 + i;
    return {
      step: index,
      padding: index < 0,
      value: index < 0 ? [0, 0, 0] : puckInput(index, side),
    };
  });
}

export function structuredCounts(rank: number) {
  if (![0, 1, 2, 4].includes(rank)) throw new Error('Only predeclared ranks are supported.');
  return { core: 2304 + 163 * rank, total: 12002 + 163 * rank, state: 64, bytes: 256 };
}

export const percentage = (value: number, digits = 1): string => (value * 100).toFixed(digits);
export const signed = (value: number): string => `${value > 0 ? '+' : ''}${value.toFixed(1)}`;
