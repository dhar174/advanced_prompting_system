/// <reference types="vitest/globals" />
import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect } from 'vitest';
import { Textarea } from './Textarea';

describe('Textarea Component', () => {
  it('displays the value prop correctly', () => {
    render(<Textarea value="Initial text" readOnly data-testid="textarea-test" />);
    expect(screen.getByTestId('textarea-test')).toHaveValue('Initial text');
  });

  it('calls onChange handler when value changes', async () => {
    const handleChange = vi.fn();
    render(<Textarea onChange={handleChange} data-testid="textarea-test" />);
    const textareaElement = screen.getByTestId('textarea-test');
    await userEvent.type(textareaElement, 'New text');
    expect(handleChange).toHaveBeenCalledTimes('New text'.length);
    // expect(textareaElement).toHaveValue('New text'); // If it were a controlled component in the test
  });

  it('renders placeholder prop', () => {
    render(<Textarea placeholder="Enter text here..." data-testid="textarea-test" />);
    expect(screen.getByPlaceholderText('Enter text here...')).toBeInTheDocument();
  });

  it('is disabled when disabled prop is true', () => {
    render(<Textarea disabled data-testid="textarea-test" />);
    expect(screen.getByTestId('textarea-test')).toBeDisabled();
  });

  it('applies the rows prop correctly', () => {
    render(<Textarea rows={5} data-testid="textarea-test" />);
    expect(screen.getByTestId('textarea-test')).toHaveAttribute('rows', '5');
  });

  it('applies default rows if rows prop is not provided', () => {
    // The component defaults to 3 rows if props.rows is not specified.
    render(<Textarea data-testid="textarea-test" />);
    expect(screen.getByTestId('textarea-test')).toHaveAttribute('rows', '3');
  });

  it('applies custom className prop', () => {
    render(<Textarea className="custom-textarea" data-testid="textarea-test" />);
    expect(screen.getByTestId('textarea-test')).toHaveClass('custom-textarea');
  });

  it('spreads other props like id and maxLength', () => {
    render(
      <Textarea
        id="my-textarea"
        maxLength={100}
        aria-label="My text area"
        data-testid="textarea-test"
      />
    );
    const textareaElement = screen.getByTestId('textarea-test');
    expect(textareaElement).toHaveAttribute('id', 'my-textarea');
    expect(textareaElement).toHaveAttribute('maxLength', '100');
    expect(textareaElement).toHaveAttribute('aria-label', 'My text area');
  });
});
