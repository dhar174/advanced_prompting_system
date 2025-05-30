import React from 'react';

export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={`flex min-h-[80px] w-full rounded-md border border-slate-300 bg-surface-card px-3 py-2 text-sm text-text-primary ring-offset-surface-ground placeholder:text-text-secondary/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring-brand focus-visible:border-ring-brand disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
        ref={ref}
        rows={props.rows || 3}
        {...props}
      />
    );
  }
);
Textarea.displayName = 'Textarea';

export { Textarea };
