import { forwardRef } from 'react';
import { Slot } from '@radix-ui/react-slot';
import { cva, type VariantProps } from 'class-variance-authority';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

/**
 * Flat pill buttons. One primary action per view uses `primary`; everything
 * else is `secondary` or `ghost`. No gradients, no glow, no lift.
 *
 * The pill is the shape that identifies this design language, so it is set on
 * the base rather than per-size, and `size="icon"` lands on a circle for free.
 * `secondary` is a *filled* neutral rather than an outline — outlines next to
 * a solid black pill read as two competing weights; a soft grey fill reads as
 * one system with the primary clearly on top.
 */
const buttonVariants = cva(
  'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-full font-medium transition-colors duration-100 disabled:pointer-events-none disabled:opacity-50 [&_svg]:shrink-0',
  {
    variants: {
      variant: {
        primary: 'bg-accent text-accent-ink hover:bg-accent-hover active:bg-ink active:text-canvas',
        secondary: 'bg-subtle text-ink hover:bg-border/60 active:bg-border',
        ghost: 'text-ink-secondary hover:bg-subtle hover:text-ink',
        danger: 'bg-danger text-white hover:bg-danger/90',
        link: 'text-ink underline-offset-4 hover:underline',
      },
      size: {
        sm: 'h-8 px-3 text-xs [&_svg]:size-3.5',
        md: 'h-10 px-4 text-sm [&_svg]:size-4',
        lg: 'h-11 px-5 text-sm [&_svg]:size-4',
        icon: 'size-10 [&_svg]:size-4',
        'icon-sm': 'size-8 [&_svg]:size-3.5',
      },
    },
    defaultVariants: { variant: 'secondary', size: 'md' },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, loading = false, children, disabled, ...props }, ref) => {
    const Component = asChild ? Slot : 'button';
    return (
      <Component
        ref={ref}
        className={cn(buttonVariants({ variant, size }), className)}
        disabled={disabled || loading}
        {...props}
      >
        {loading ? <Loader2 className="animate-spin" aria-hidden /> : null}
        {children}
      </Component>
    );
  },
);
Button.displayName = 'Button';

export { buttonVariants };
