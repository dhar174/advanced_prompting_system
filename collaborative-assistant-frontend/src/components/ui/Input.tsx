import React from 'react';

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={`flex h-10 w-full rounded-md border border-slate-300 bg-surface-card px-3 py-2 text-sm text-text-primary ring-offset-surface-ground file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-text-secondary/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring-brand focus-visible:border-ring-brand disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = 'Input';

export { Input };
