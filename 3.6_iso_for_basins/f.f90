subroutine A(sos)
real*8 sos
sos=sos+5
end subroutine

program B
real*8 :: i=5
call A(i)
write(*,*) i
end program 


