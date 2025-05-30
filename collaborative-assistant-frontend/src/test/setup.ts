// src/test/setup.ts
import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Mock scrollIntoView as it's not implemented in JSDOM
Element.prototype.scrollIntoView = vi.fn();
