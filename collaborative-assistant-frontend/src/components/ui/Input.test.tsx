/// <reference types="vitest/globals" />
import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect } from 'vitest';
import { Input } from './Input';

describe('Input Component', () => {
  it('renders with different type props', () => {
    const { rerender } = render(<Input type="text" data-testid="input-test" />);
    expect(screen.getByTestId('input-test')).toHaveAttribute('type', 'text');

    rerender(<Input type="password" data-testid="input-test" />);
    expect(screen.getByTestId('input-test')).toHaveAttribute('type', 'password');

    rerender(<Input type="number" data-testid="input-test" />);
    expect(screen.getByTestId('input-test')).toHaveAttribute('type', 'number');
  });

  it('displays the value prop correctly', () => {
    render(<Input type="text" value="Test Value" readOnly data-testid="input-test" />);
    expect(screen.getByTestId('input-test')).toHaveValue('Test Value');
  });

  it('calls onChange handler when value changes', async () => {
    const handleChange = vi.fn();
    render(<Input type="text" onChange={handleChange} data-testid="input-test" />);
    const inputElement = screen.getByTestId('input-test');
    await userEvent.type(inputElement, 'Hello');
    expect(handleChange).toHaveBeenCalledTimes(5); // Called for each character 'H', 'e', 'l', 'l', 'o'
    // More specific check for the final value if needed, though React handles state typically
    // expect(inputElement).toHaveValue('Hello'); // This would be true if it was a controlled component in the test
  });

  it('renders placeholder prop', () => {
    render(<Input type="text" placeholder="Enter text..." data-testid="input-test" />);
    expect(screen.getByPlaceholderText('Enter text...')).toBeInTheDocument();
  });

  it('is disabled when disabled prop is true', () => {
    render(<Input type="text" disabled data-testid="input-test" />);
    expect(screen.getByTestId('input-test')).toBeDisabled();
  });

  it('applies custom className prop', () => {
    render(<Input type="text" className="custom-input" data-testid="input-test" />);
    expect(screen.getByTestId('input-test')).toHaveClass('custom-input');
  });

  it('spreads other props like id and maxLength', () => {
    render(
      <Input
        type="text"
        id="my-input"
        maxLength={10}
        aria-label="My text input"
        data-testid="input-test"
      />
    );
    const inputElement = screen.getByTestId('input-test');
    expect(inputElement).toHaveAttribute('id', 'my-input');
    expect(inputElement).toHaveAttribute('maxLength', '10');
    expect(inputElement).toHaveAttribute('aria-label', 'My text input');
  });
});
