import raw from '../../public/data/evidence.json';
export type Family =
  | 'feed_forward'
  | 'finite_stack_10'
  | 'structured_k0'
  | 'structured_k1'
  | 'structured_k2'
  | 'structured_k4'
  | 'gru_n64';
export const families: Family[] = [
  'feed_forward',
  'finite_stack_10',
  'structured_k0',
  'structured_k1',
  'structured_k2',
  'structured_k4',
  'gru_n64',
];
export const labels: Record<Family | 'teacher', string> = {
  feed_forward: 'Feed-forward',
  finite_stack_10: 'Ten-step stack',
  structured_k0: 'Structured k=0',
  structured_k1: 'Structured k=1',
  structured_k2: 'Structured k=2',
  structured_k4: 'Structured k=4',
  gru_n64: 'GRU-64',
  teacher: 'Teacher',
};
export const colours: Record<Family | 'teacher', string> = {
  feed_forward: '#6f7275',
  finite_stack_10: '#9a6500',
  structured_k0: '#007d64',
  structured_k1: '#0076a8',
  structured_k2: '#a96191',
  structured_k4: '#bb512c',
  gru_n64: '#605197',
  teacher: '#222d28',
};
export const dashes: Record<Family | 'teacher', string> = {
  feed_forward: '',
  finite_stack_10: '6 3',
  structured_k0: '',
  structured_k1: '8 3',
  structured_k2: '2 3',
  structured_k4: '9 3 2 3',
  gru_n64: '4 2',
  teacher: '2 5',
};
export const evidence = raw;
export const principal = raw.principal as unknown as {
  student_save_rates: Record<Family, Record<string, number>>;
  student_save_rate_95_intervals: Record<Family, Record<string, number[]>>;
  student_seed_save_rates: Record<Family, Record<string, Record<string, number>>>;
  teacher_save_rates: Record<string, number>;
  planned_differences_at_20_steps: Record<
    string,
    {
      left: Family;
      right: Family;
      estimate_points: number;
      percentile_95_interval_points: number[];
    }
  >;
  efficiency_seed_points: Record<
    Family,
    { median_microseconds: number; p95_microseconds: number; training_seed: number }[]
  >;
};
export const durations = [0, 5, 10, 15, 20, 25];
export const rate = (family: Family | 'teacher', step: number) =>
  family === 'teacher'
    ? principal.teacher_save_rates[step]
    : principal.student_save_rates[family][step];
export const interval = (family: Family, step: number) =>
  principal.student_save_rate_95_intervals[family][step];
export const median = (values: number[]) =>
  [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
