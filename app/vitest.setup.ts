import '@testing-library/jest-dom/vitest';

// Suppress React act warnings during tests
const originalError = console.error;
beforeAll(() => {
  vi.spyOn(console, 'error').mockImplementation((...args) => {
    if (typeof args[0] === 'string' && args[0].includes('not wrapped in act')) {
      return;
    }
    originalError(...args);
  });
});

afterAll(() => {
  (console.error as any).mockRestore();
});
