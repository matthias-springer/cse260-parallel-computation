	.file	"dgemm-blocked.c"
	.text
	.p2align 4,,15
	.globl	square_dgemm
	.type	square_dgemm, @function
square_dgemm:
.LFB1:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	testl	%edi, %edi
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rsi, -48(%rsp)
	movq	%rdx, -40(%rsp)
	movq	%rcx, -32(%rsp)
	jle	.L1
	movl	%edi, %eax
	movslq	%edi, %rsi
	movl	%edi, -84(%rsp)
	sall	$5, %eax
	salq	$3, %rsi
	movq	$0, -72(%rsp)
	cltq
	movl	%edi, -52(%rsp)
	salq	$3, %rax
	movq	%rax, -64(%rsp)
	leal	-32(%rdi), %eax
	movl	%eax, -56(%rsp)
	leal	-1(%rdi), %eax
	andl	$-32, %eax
	subl	%eax, -56(%rsp)
.L3:
	cmpl	$32, -84(%rsp)
	movq	-48(%rsp), %rdx
	movl	$32, %r15d
	movq	-72(%rsp), %rax
	cmovle	-84(%rsp), %r15d
	movq	-32(%rsp), %rcx
	addq	-72(%rsp), %rcx
	movl	-52(%rsp), %edi
	movq	$0, -80(%rsp)
	leaq	(%rdx,%rax), %rax
	movl	%edi, -92(%rsp)
	movq	%rax, -24(%rsp)
	movq	%rcx, -8(%rsp)
.L12:
	cmpl	$32, -92(%rsp)
	movl	$32, %r12d
	movq	-40(%rsp), %rdi
	cmovle	-92(%rsp), %r12d
	movq	-80(%rsp), %rcx
	movq	-24(%rsp), %rbp
	leaq	8(%rdi,%rcx), %rdx
	movq	%rcx, %r13
	movl	-52(%rsp), %edi
	leal	-1(%r12), %eax
	addq	-8(%rsp), %r13
	leaq	(%rdx,%rax,8), %r10
	negq	%rax
	movl	%edi, -88(%rsp)
	leaq	-8(,%rax,8), %rax
	movq	%rax, -16(%rsp)
.L10:
	cmpl	$32, -88(%rsp)
	movl	$32, %ecx
	cmovle	-88(%rsp), %ecx
	testl	%r15d, %r15d
	jle	.L4
	movq	-16(%rsp), %r14
	xorl	%r11d, %r11d
	xorl	%ebx, %ebx
	addq	%r10, %r14
.L5:
	testl	%r12d, %r12d
	jle	.L9
	leaq	(%r11,%r13), %r8
	leaq	(%r11,%rbp), %rdi
	movq	%r14, %r9
	.p2align 4,,10
	.p2align 3
.L8:
	testl	%ecx, %ecx
	movsd	(%r8), %xmm1
	jle	.L6
	movq	%r9, %rdx
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L7:
	movsd	(%rdi,%rax,8), %xmm0
	addq	$1, %rax
	mulsd	(%rdx), %xmm0
	addq	%rsi, %rdx
	cmpl	%eax, %ecx
	addsd	%xmm0, %xmm1
	jg	.L7
.L6:
	addq	$8, %r9
	movsd	%xmm1, (%r8)
	addq	$8, %r8
	cmpq	%r9, %r10
	jne	.L8
.L9:
	addl	$1, %ebx
	addq	%rsi, %r11
	cmpl	%r15d, %ebx
	jne	.L5
.L4:
	subl	$32, -88(%rsp)
	addq	-64(%rsp), %r10
	addq	$256, %rbp
	movl	-56(%rsp), %edx
	cmpl	%edx, -88(%rsp)
	jne	.L10
	subl	$32, -92(%rsp)
	addq	$256, -80(%rsp)
	cmpl	%edx, -92(%rsp)
	jne	.L12
	subl	$32, -84(%rsp)
	movq	-64(%rsp), %rax
	movl	-56(%rsp), %edx
	addq	%rax, -72(%rsp)
	cmpl	%edx, -84(%rsp)
	jne	.L3
.L1:
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE1:
	.size	square_dgemm, .-square_dgemm
	.globl	dgemm_desc
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"Simple blocked dgemm."
	.data
	.align 8
	.type	dgemm_desc, @object
	.size	dgemm_desc, 8
dgemm_desc:
	.quad	.LC0
	.ident	"GCC: (GNU) 4.7.3"
	.section	.note.GNU-stack,"",@progbits
