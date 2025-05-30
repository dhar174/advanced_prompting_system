import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Button } from './Button';
import { describe, it, expect, vi } from 'vitest';

describe('Button component', () => {
  it('renders children correctly', () => {
    render(<Button>Click Me</Button>);
    expect(screen.getByText('Click Me')).toBeInTheDocument();
  });

  it('applies primary variant by default', () => {
    render(<Button>Primary Button</Button>);
    expect(screen.getByText('Primary Button')).toHaveClass('bg-brand-primary');
  });

  it('applies secondary variant correctly', () => {
    render(<Button variant="secondary">Secondary Button</Button>);
    expect(screen.getByText('Secondary Button')).toHaveClass('bg-brand-secondary');
  });

  it('applies danger variant correctly', () => {
    render(<Button variant="danger">Danger Button</Button>);
    expect(screen.getByText('Danger Button')).toHaveClass('bg-red-600');
  });

  it('applies outline variant correctly', () => {
    render(<Button variant="outline">Outline Button</Button>);
    expect(screen.getByText('Outline Button')).toHaveClass('border-brand-primary');
  });

  it('applies ghost variant correctly', () => {
    render(<Button variant="ghost">Ghost Button</Button>);
    // Ghost might not have a specific class, but hover state is key
    // For now, just check it renders
    expect(screen.getByText('Ghost Button')).toBeInTheDocument();
  });

  it('applies size classes correctly', () => {
    render(<Button size="sm">Small</Button>);
    expect(screen.getByText('Small')).toHaveClass('h-9');
    render(<Button size="lg">Large</Button>);
    // Need to get by role or other means if text changes
    expect(screen.getByText('Large')).toHaveClass('h-11');
  });

  it('is disabled when disabled prop is true', () => {
    render(<Button disabled>Disabled Button</Button>);
    expect(screen.getByText('Disabled Button')).toBeDisabled();
    expect(screen.getByText('Disabled Button')).toHaveClass('disabled:opacity-50');
  });

  it('calls onClick handler when clicked', async () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick}>Clickable</Button>);
    await userEvent.click(screen.getByText('Clickable'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('does not call onClick handler when disabled and clicked', async () => {
    const handleClick = vi.fn();
    render(<Button onClick={handleClick} disabled>Disabled Click</Button>);
    // userEvent.click on a disabled button should not throw, but also not call handler
    // fireEvent.click might be more direct for testing if userEvent has issues with disabled
    // However, userEvent aims to simulate real user behavior more closely.
    
    // Attempt click
    await userEvent.click(screen.getByText('Disabled Click'));
    
    // Check if handler was called
    expect(handleClick).not.toHaveBeenCalled();
  });

   it('applies custom className', () => {
    render(<Button className="custom-class">Custom</Button>);
    expect(screen.getByText('Custom')).toHaveClass('custom-class');
  });
});
