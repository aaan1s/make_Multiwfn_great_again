Module Zhopa
	implicit none
contains
	!!----- Make the matrix to upper trigonal matrix
	subroutine ratio_upper(mat)
	real*8 :: mat(:,:),m,st
	real*8,allocatable :: temp(:),s(:),divided(:)
	integer :: a,i,j,t
	a=size(mat,1)
	allocate(temp(a))
	allocate(s(a))
	allocate(divided(a))
	do i=1,a
		s(i)=maxval(abs(mat(i,1:a)))
	end do
	do i=1,a-1
		divided(i:a)=mat(i:a,i)/s(i:a)
		t=maxloc(abs(divided(i:a)),dim=1)
		temp(:)=mat(i,:)
		mat(i,:)=mat(i+t-1,:)
		mat(i+t-1,:)=temp(:)
		st=s(i)
		s(i)=s(i+t-1)
		s(i+t-1)=st
		do j=i+1,a
			m=mat(j,i)/mat(i,i)
			mat(j,i:a)=mat(j,i:a)-mat(i,i:a)*m
		end do
	end do
	deallocate(temp,s,divided)
	end subroutine


	!!----- Get value of determinant of a matrix
	real*8 function detmat(mat)
	real*8 mat(:,:)
	real*8,allocatable :: mattmp(:,:)
	integer :: NOTlowertri, NOTuppertri, i, j, isizemat
        isizemat=size(mat,1)
	detmat=1D0
	NOTlowertri=0
	NOTuppertri=0
	
	outter1: do i=1,isizemat !Check if already is lower-trigonal matrix
		do j=i+1,isizemat
			if (mat(i,j)>1D-12) then
				NOTlowertri=1 !There are at least one big value at upper trigonal part, hence not lower trigonal matrix
				exit outter1
			end if
		end do
	end do outter1
	outter2: do i=1,isizemat !Check if already is upper-trigonal matrix
		do j=1,i-1
			if (mat(i,j)>1D-12) then
				NOTuppertri=1 !There are at least one big value at lower trigonal part, hence not upper trigonal matrix
				exit outter2
			end if
		end do
	end do outter2
	
	if (NOTlowertri==0.or.NOTuppertri==0) then !Is lower or upper trigonal matrix, don't need to convert to trigonal matrix
		do i=1,isizemat
			detmat=detmat*mat(i,i)
		end do
	else !Not upper or lower trigonal matrix
		allocate(mattmp(isizemat,isizemat))
		mattmp=mat
		call ratio_upper(mattmp)
		detmat=1D0
		do i=1,isizemat
			detmat=detmat*mattmp(i,i)
		end do
	end if
	end function
end Module Zhopa

program main
    use zhopa
    implicit none
    real*8 b
    real*8 a(3,3)
    a = reshape((/ 35912.7743047545_8,        37561.1544582623_8,       2.244047572193334D-012, &
    37561.1544582623_8,        35912.7743047545_8,       2.253335344568352D-012, &
    2.244047572193334D-012,  2.253335344568352D-012,   211.200880461918_8 /), (/3,3/))    
    !print *,a  
    b=detmat(a)
    print *,b
end program main
