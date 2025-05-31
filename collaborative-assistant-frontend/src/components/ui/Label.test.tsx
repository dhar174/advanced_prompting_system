/// <reference types="vitest/globals" />
import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { Label } from './Label'; // Assuming Label is exported from './Label'

describe('Label Component', () => {
  it('renders its children as label text', () => {
    render(<Label>This is a label</Label>);
    expect(screen.getByText('This is a label')).toBeInTheDocument();
  });

  it('applies custom className prop', () => {
    render(<Label className="custom-label">Label text</Label>);
    expect(screen.getByText('Label text')).toHaveClass('custom-label');
  });

  it('passes htmlFor prop correctly', () => {
    render(<Label htmlFor="inputId">Label for input</Label>);
    // Radix UI Label renders a <label> element.
    // We can find it by text and check its 'for' attribute.
    expect(screen.getByText('Label for input')).toHaveAttribute('for', 'inputId');
  });

  it('spreads other props like id and aria-details', () => {
    render(
      <Label id="my-label" aria-details="details-for-label">
        Label text
      </Label>
    );
    const labelElement = screen.getByText('Label text');
    expect(labelElement).toHaveAttribute('id', 'my-label');
    expect(labelElement).toHaveAttribute('aria-details', 'details-for-label');
  });

  // Test for peer-disabled opacity (visual test, hard to assert computed style reliably without more setup)
  // For now, we trust Radix and class application. If specific styles were critical,
  // one might use getComputedStyle or snapshot testing with style serialization.
  it('applies base variant classes', () => {
    render(<Label>Test Label</Label>);
    // Check for one of the default classes from labelVariants
    expect(screen.getByText('Test Label')).toHaveClass('text-sm');
  });
});
