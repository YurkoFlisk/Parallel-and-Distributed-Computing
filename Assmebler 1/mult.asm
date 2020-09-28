.CODE

mat_dot_prod_64_masm_nommx PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix

    ; rbx - index
    ; rdi - sum

    xor rbx, rbx
    xor rdi, rdi
    xor rax, rax
    
next:
    mov al, [rcx+rbx]
    mul byte ptr [rdx+rbx] ; ax = al * memory
    add edi, eax
    inc rbx
    cmp rbx, 64
    jl next
    
    mov eax, edi
    ret
mat_dot_prod_64_masm_nommx ENDP

mat_dot_prod_64_masm_mmx PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix

    ; rbx - index (of current iteration)
    ; mm0 - constant zero
    ; mm5 - sum
    
    xor rbx, rbx
    pxor mm0, mm0
    pxor mm5, mm5

next:
    movq mm1, [rcx+8*rbx]
    movq mm2, mm1
    punpckhbw mm1, mm0
    punpcklbw mm2, mm0

    movq mm3, [rdx+8*rbx]
    movq mm4, mm3
    punpckhbw mm3, mm0
    punpcklbw mm4, mm0

    pmaddwd mm1, mm3
    pmaddwd mm2, mm4
    paddd mm1, mm2
    paddd mm5, mm1
    
    inc rbx
    cmp rbx, 8 ; 64/8, no left for tail
    jl next
    
    movd eax, mm5 ; extract lower 32-bit of mm5 into eax 
    psrlq mm5, 32 ; extract higher 32-but of mm5 into ebx
    movd ebx, mm5 ; ..
    add eax, ebx  ; add both sums
    ret
mat_dot_prod_64_masm_mmx ENDP

mat_dot_prod_gen_masm_nommx PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix
    ; r8  - count of elements

    ; rbx - index
    ; rdi - sum register

    xor rbx, rbx
    xor rdi, rdi
    xor rax, rax

    cmp r8, 0 ; zero elements case 
    jnz next
    ret

next:
    mov al, [rcx+rbx]
    mul byte ptr [rdx+rbx] ; ax = al * memory
    add edi, eax
    inc rbx
    cmp rbx, r8
    jl next
    
    mov eax, edi
    ret
mat_dot_prod_gen_masm_nommx ENDP

mat_dot_prod_gen_masm_mmx PROC
    ; PARAMS:
    ; rcx - first matrix
    ; rdx - second matrix
    ; r8  - count of elements

    ; rbx - index (of iteration)
    ; mm0 - constant zero
    ; mm5 - sum register

    xor rbx, rbx
    pxor mm0, mm0
    pxor mm5, mm5

    mov r9, r8
    shr r9, 3
    jz tail_pre

next:
    movq mm1, [rcx+8*rbx]
    movq mm2, mm1
    punpckhbw mm1, mm0
    punpcklbw mm2, mm0

    movq mm3, [rdx+8*rbx]
    movq mm4, mm3
    punpckhbw mm3, mm0
    punpcklbw mm4, mm0

    pmaddwd mm1, mm3
    pmaddwd mm2, mm4
    paddd mm1, mm2
    paddd mm5, mm1
    
    inc rbx
    cmp rbx, r9
    jl next

    ; now rbx = r9
    shl rbx, 3 
tail_pre:
    cmp rbx, r8
    jge exit

    xor rax, rax
    xor rdi, rdi
tail:
    mov al, [rcx+rbx]
    mul byte ptr [rdx+rbx] ; ax = al * memory
    add edi, eax

    inc rbx
    cmp rbx, r8
    jl tail

exit:
    movd eax, mm5 ; extract lower 32-bit of mm5 into eax 
    psrlq mm5, 32 ; extract higher 32-but of mm5 into ebx
    movd ebx, mm5 ; ..
    add eax, ebx  ; add both sums
    add eax, edi  ; add tail sum
    ret
mat_dot_prod_gen_masm_mmx ENDP

end