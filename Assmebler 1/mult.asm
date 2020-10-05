.CODE

; 64 VALUES VARIANTS

mat_dot_prod_64_masm_nommx PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix

    ; r10 - index
    ; r11 - sum

    xor r10, r10
    xor r11, r11
    xor rax, rax
    
next:
    mov al, [rcx+r10]
    mul byte ptr [rdx+r10] ; ax = al * memory
    add r11d, eax
    inc r10
    cmp r10, 64
    jl next
    
    mov eax, r11d
    ret
mat_dot_prod_64_masm_nommx ENDP

mat_dot_prod_64_masm_mmx PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix

    ; r10 - index (of current iteration)
    ; mm0 - constant zero
    ; mm5 - sum
    
    xor r10, r10
    pxor mm0, mm0
    pxor mm5, mm5

next:
    movq mm1, [rcx+8*r10]
    movq mm2, mm1
    punpckhbw mm1, mm0
    punpcklbw mm2, mm0

    movq mm3, [rdx+8*r10]
    movq mm4, mm3
    punpckhbw mm3, mm0
    punpcklbw mm4, mm0

    pmaddwd mm1, mm3
    pmaddwd mm2, mm4
    paddd mm1, mm2
    paddd mm5, mm1
    
    inc r10
    cmp r10, 8 ; 64/8, no left for tail
    jl next
    
    movd eax, mm5 ; extract lower 32-bit of mm5 into eax 
    psrlq mm5, 32 ; extract higher 32-but of mm5 into r10d
    movd r10d, mm5 ; ..
    add eax, r10d  ; add both sums
    ret
mat_dot_prod_64_masm_mmx ENDP

mat_dot_prod_64_masm_sse_32bitsum PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix

    ; r10 - index (of current iteration)
    ; xmm0 - constant zero
    ; xmm5 - sum
    
    xor r10, r10
    pxor xmm0, xmm0
    pxor xmm5, xmm5

next:
    movdqa xmm1, [rcx+r10]
    movdqa xmm2, xmm1
    punpckhbw xmm1, xmm0
    punpcklbw xmm2, xmm0

    movdqa xmm3, [rdx+r10]
    movdqa xmm4, xmm3
    punpckhbw xmm3, xmm0
    punpcklbw xmm4, xmm0
    
    pmaddwd xmm1, xmm3
    pmaddwd xmm2, xmm4
    paddd xmm1, xmm2
    paddd xmm5, xmm1
    
    add r10, 16
    cmp r10, 64 ; no left for tail
    jl next
    
    phaddd xmm5, xmm0
    phaddd xmm5, xmm0
    pextrd eax, xmm5, 0
    ret
mat_dot_prod_64_masm_sse_32bitsum ENDP

mat_dot_prod_64_masm_sse_16bitsum PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix

    ; r10 - index (of current iteration)
    ; xmm0 - constant zero
    ; xmm5 - sum
    
    xor r10, r10
    pxor xmm0, xmm0
    pxor xmm5, xmm5

next:
    movdqa xmm1, [rcx+r10]
    movdqa xmm3, [rdx+r10]

    pmaddubsw  xmm1, xmm3
    paddw xmm5, xmm1
    
    add r10, 16
    cmp r10, 64 ; no left for tail
    jl next
    
    phaddw xmm5, xmm0
    phaddw xmm5, xmm0
    phaddw xmm5, xmm0
    pextrw eax, xmm5, 0
    ret
mat_dot_prod_64_masm_sse_16bitsum ENDP

; ARBITRARY VALUES VARIANTS

mat_dot_prod_gen_masm_nommx PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix
    ; r8  - count of elements

    ; r10 - index
    ; r11 - sum register

    xor r10, r10
    xor r11, r11
    xor rax, rax

    cmp r8, 0 ; zero elements case 
    jnz next
    ret

next:
    mov al, [rcx+r10]
    mul byte ptr [rdx+r10] ; ax = al * memory
    add r11d, eax
    inc r10
    cmp r10, r8
    jl next
    
    mov eax, r11d
    ret
mat_dot_prod_gen_masm_nommx ENDP

mat_dot_prod_gen_masm_mmx PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix
    ; r8  - count of elements

    ; r10 - index (of iteration)
    ; r11 - sum of the tail
    ; mm0 - constant zero
    ; mm5 - sum register

    xor r10, r10
    xor r11, r11
    pxor mm0, mm0
    pxor mm5, mm5

    mov r9, r8
    shr r9, 3
    jz tail_pre

next:
    movq mm1, [rcx+8*r10]
    movq mm2, mm1
    punpckhbw mm1, mm0
    punpcklbw mm2, mm0

    movq mm3, [rdx+8*r10]
    movq mm4, mm3
    punpckhbw mm3, mm0
    punpcklbw mm4, mm0

    pmaddwd mm1, mm3
    pmaddwd mm2, mm4
    paddd mm1, mm2
    paddd mm5, mm1
    
    inc r10
    cmp r10, r9
    jl next

    ; now r10 = r9
    shl r10, 3 
tail_pre:
    cmp r10, r8
    jge exit

    xor rax, rax
tail:
    mov al, [rcx+r10]
    mul byte ptr [rdx+r10] ; ax = al * memory
    add r11d, eax

    inc r10
    cmp r10, r8
    jl tail

exit:
    movd eax, mm5 ; extract lower 32-bit of mm5 into eax 
    psrlq mm5, 32 ; extract higher 32-but of mm5 into r10d
    movd r10d, mm5 ; ..
    add eax, r10d  ; add both sums
    add eax, r11d  ; add tail sum
    ret
mat_dot_prod_gen_masm_mmx ENDP

mat_dot_prod_gen_masm_sse_32bitsum PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix
    ; r8  - count of elements

    ; r10 - index (of the current value)
    ; r11 - sum of the tail
    ; xmm0 - constant zero
    ; xmm5 - sum register

    xor r10, r10
    xor r11, r11
    pxor xmm0, xmm0
    pxor xmm5, xmm5

    mov r9, r8
    shr r9, 4
    shl r9, 4 ; first index not processed by main loop
    jz tail_pre

next:
    movdqa xmm1, [rcx+r10]
    movdqa xmm2, xmm1
    punpckhbw xmm1, xmm0
    punpcklbw xmm2, xmm0

    movdqa xmm3, [rdx+r10]
    movdqa xmm4, xmm3
    punpckhbw xmm3, xmm0
    punpcklbw xmm4, xmm0
    
    pmaddwd xmm1, xmm3
    pmaddwd xmm2, xmm4
    paddd xmm1, xmm2
    paddd xmm5, xmm1
    
    add r10, 16
    cmp r10, r9
    jl next

    ; now r10 = r9
tail_pre:
    cmp r10, r8
    jge exit

    xor rax, rax
tail:
    mov al, [rcx+r10]
    mul byte ptr [rdx+r10] ; ax = al * memory
    add r11d, eax

    inc r10
    cmp r10, r8
    jl tail

exit:
    phaddd xmm5, xmm0
    phaddd xmm5, xmm0
    pextrd eax, xmm5, 0
    add eax, r11d ; add tail sum
    ret
mat_dot_prod_gen_masm_sse_32bitsum ENDP

mat_dot_prod_gen_masm_sse_16bitsum PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix
    ; r8  - count of elements

    ; r10 - index (of iteration)
    ; r11 - sum of the tail
    ; xmm0 - constant zero
    ; xmm5 - sum register

    xor r10, r10
    xor r11, r11
    pxor xmm0, xmm0
    pxor xmm5, xmm5

    mov r9, r8
    shr r9, 4
    shl r9, 4 ; first index not processed by main loop
    jz tail_pre

next:
    movdqa xmm1, [rcx+r10]
    movdqa xmm3, [rdx+r10]

    pmaddubsw xmm1, xmm3
    paddw xmm5, xmm1
    
    add r10, 16
    cmp r10, r9
    jl next

    ; now r10 = r9
tail_pre:
    cmp r10, r8
    jge exit

    xor rax, rax
tail:
    mov al, [rcx+r10]
    mul byte ptr [rdx+r10] ; ax = al * memory
    add r11w, ax

    inc r10
    cmp r10, r8
    jl tail

exit:
    phaddw xmm5, xmm0
    phaddw xmm5, xmm0
    phaddw xmm5, xmm0
    pextrw eax, xmm5, 0
    add ax, r11w ; add tail sum
    ret
mat_dot_prod_gen_masm_sse_16bitsum ENDP

end