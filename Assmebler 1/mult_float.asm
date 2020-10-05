.CODE

; 64 VALUES VARIANTS

matf_dot_prod_64_masm_sse PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix

    ; r10 - index (of current element)
    ; xmm0 - sum
    
    xor r10, r10
    pxor xmm0, xmm0

next:
    movaps xmm1, [rcx+4*r10]
    movaps xmm3, [rdx+4*r10]
    
    dpps xmm1, xmm3, 0F1h
    addss xmm0, xmm1
    
    add r10, 4
    cmp r10, 64 ; no left for tail
    jl next
    
    ret
matf_dot_prod_64_masm_sse ENDP

matf_dot_prod_64_masm_sse_mulps PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix

    ; r10 - index (of current element)
    ; xmm0 - sum
    ; xmm5 - zero
    
    xor r10, r10
    pxor xmm0, xmm0
    pxor xmm5, xmm5

next:
    movaps xmm1, [rcx+4*r10]
    mulps xmm1, [rdx+4*r10]
    addps xmm0, xmm1
    
    add r10, 4
    cmp r10, 64 ; no left for tail
    jl next
    
    haddps xmm0, xmm5
    haddps xmm0, xmm5

    ret
matf_dot_prod_64_masm_sse_mulps ENDP

; ARBITRARY VALUES VARIANTS

matf_dot_prod_gen_masm_sse PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix
    ; r8  - count of elements

    ; r10 - index (of iteration)
    ; xmm0 - sum

    xor r10, r10
    pxor xmm0, xmm0

    mov r9, r8
    shr r9, 2
    shl r9, 2 ; first index not processed by main loop
    jz tail_pre

next:
    movaps xmm1, [rcx+4*r10]
    movaps xmm3, [rdx+4*r10]

    dpps xmm1, xmm3, 0F1h
    addss xmm0, xmm1
    
    add r10, 4
    cmp r10, r9
    jl next
    
    ; now r10 = r9
tail_pre:
    cmp r10, r8
    jge exit

tail:
    movss xmm1, DWORD PTR [rcx+4*r10]
    mulss xmm1, DWORD PTR [rdx+4*r10]
    addss xmm0, xmm1

    inc r10
    cmp r10, r8
    jl tail

exit:
    ret
matf_dot_prod_gen_masm_sse ENDP

matf_dot_prod_gen_masm_sse_mulps PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix
    ; r8  - count of elements

    ; r10 - index (of iteration)
    ; xmm0 - sum
    ; xmm5 - zero

    xor r10, r10
    pxor xmm0, xmm0
    pxor xmm5, xmm5

    mov r9, r8
    shr r9, 2
    shl r9, 2 ; first index not processed by main loop
    jz tail_pre

next:
    movaps xmm1, [rcx+4*r10]
    mulps xmm1, [rdx+4*r10]
    addps xmm0, xmm1
    
    add r10, 4
    cmp r10, r9
    jl next
    
    ; now r10 = r9
    haddps xmm0, xmm5
    haddps xmm0, xmm5
tail_pre:
    cmp r10, r8
    jge exit

tail:
    movss xmm1, DWORD PTR [rcx+4*r10]
    mulss xmm1, DWORD PTR [rdx+4*r10]
    addss xmm0, xmm1

    inc r10
    cmp r10, r8
    jl tail

exit:
    ret
matf_dot_prod_gen_masm_sse_mulps ENDP

end