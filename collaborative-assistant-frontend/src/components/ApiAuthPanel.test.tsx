/// <reference types="vitest/globals" />
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import ApiAuthPanel from './ApiAuthPanel';

describe('ApiAuthPanel', () => {
  it('renders masked token input, accessible labels, and submits/clears tokens', async () => {
    const onApplyToken = vi.fn();
    const onClearToken = vi.fn();

    const { rerender } = render(
      <ApiAuthPanel
        authRequired={false}
        hasToken={false}
        onApplyToken={onApplyToken}
        onClearToken={onClearToken}
      />
    );

    const input = screen.getByLabelText('API Bearer Token') as HTMLInputElement;
    expect(input.type).toBe('password');
    expect(screen.getByTestId('apply-api-token-button')).toBeDisabled();
    expect(screen.getByTestId('clear-api-token-button')).toBeDisabled();

    await userEvent.type(input, '  my-token-123  ');
    expect(screen.getByTestId('apply-api-token-button')).toBeEnabled();
    expect(screen.getByTestId('clear-api-token-button')).toBeEnabled();

    await userEvent.click(screen.getByRole('button', { name: 'Show API token' }));
    expect(input.type).toBe('text');

    await userEvent.click(screen.getByTestId('apply-api-token-button'));
    expect(onApplyToken).toHaveBeenCalledWith('my-token-123');

    expect(input.value).toBe('');
    expect(input.type).toBe('password');
    rerender(<ApiAuthPanel authRequired={false} hasToken={true} onApplyToken={onApplyToken} onClearToken={onClearToken} />);
    await userEvent.click(screen.getByTestId('clear-api-token-button'));
    expect(onClearToken).toHaveBeenCalledTimes(1);
    expect(input.value).toBe('');
    expect(input.type).toBe('password');
  });

  it('displays accessible alert and marks input invalid when authRequired is true', () => {
    render(
      <ApiAuthPanel
        authRequired={true}
        hasToken={false}
        errorMessage="Invalid bearer token."
        onApplyToken={vi.fn()}
        onClearToken={vi.fn()}
      />
    );

    expect(screen.getByRole('alert')).toHaveTextContent('Invalid bearer token.');
    expect(screen.getByLabelText('API Bearer Token')).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByTestId('apply-api-token-button')).toHaveTextContent('Save & Retry');
  });
});
