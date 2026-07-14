export function add(a: number, b: number): number {
  return a + b;
}

export function subtract(a: number, b: number): number {
  return a - b;
}

export function multiply(a: number, b: number): number {
  return a * b;
}

export function divide(a: number, b: number): number {
  if (b === 0) {
    throw new Error('Division by zero');
  }
  return a / b;
}

export function power(base: number, exp: number): number {
  if (exp < 0) {
    throw new Error('Negative exponent not supported');
  }
  if (exp === 0) {
    return 1;
  }
  let result = 1;
  for (let i = 0; i < exp; i++) {
    result *= base;
  }
  return result;
}
