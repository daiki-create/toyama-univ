
#define FORTRAN
#include "hypdet_3d.gfortran/hypdet.h"

      character buf100*100
      real::flat_t(500),flon_t(500)
      integer:: istno_t(500)
      data isz_st_t/500/
!-----------------------------------------------------------------
      data isz_th,isz_tst/200,300/
      integer:: ith_time(7,200)
      real:: th_lat(200),th_lon(200),th_dep(200),th_mag(200),th_mw(200)
      integer:: ntst(200),ist_no(200,300)
      real:: tst_lat(200,300),tst_lon(200,300)
      real:: tst_high(200,300)
      real:: pstlat(300), pstlon(300),psthigh(300)
!-----------------------------------------------------------------
!   for map
      CHARACTER*1  MAPK,mapkwk
      CHARACTER*25 CMNT
!  PLTPEN  PENNUMBER
      COMMON /PLTPEN/ IPFRAM,IPLLL,IPSCL,IPSHOR,IPPREF,IPCRSS, &
     &                IPCONT,IP76,IP67,IPOTH,IPHYPS
!  PLTFRM  FRAME INFORMATION
      COMMON /PLTFRM/ XL,YL,X0,Y0,XOFF,YOFF,SCL
!  PLTLLI   LATITUDE & LONGITUDE INFORMATION
      COMMON  /PLTLLI/  FLAT0,FLON0,FLATIN,FLONIN, &
     &                  FRLAT,TOLAT,FRLON,TOLON,FRLOND,TOLOND, &
     &                  MTHD,THETA
!  PLTCMT  COMMENT
      COMMON /PLTCMT/ CMNT,MAPK
!-----------------------------------------------------------------
      real trp,trs ! pp/ss 
      common /pp/trp(DISDIM,DEPDIM,2)
      common /ss/trs(DISDIM,DEPDIM,2)
      integer nllst,llst ! llst
      common /llst/nllst,llst(1000)
!------ station info.---------------------------------------------
      character ntat*6,jntat*8 ! m2
      common /m2/ntat(MAXS),jntat(MAXS)
      real*8  as,bs,cs,sttlat,sttlon,stthei ! m1
      integer kankn                  ! m1
      common /m1/as(MAXS),bs(MAXS),cs(MAXS),  &
                 sttlat(MAXS),sttlon(MAXS),stthei(MAXS),  &
                 kankn(MAXS)
!---------------------------------------------------------------
      character fname(50000)*80,fname_ps*50,fname_out*50,fname_out2*50
      dimension ihnum(50000)
      dimension fstlat(50000),fstlon(50000)
      dimension dlt_sort(50000), idx_sort(50000), idx_file(50000),hval_tsunami(50000)
      dimension iwv(120000)
      real*8 wvdata(120000)
      data isz_wv/120000/
      dimension ihtime(7,600),iwvtime(7),iwhtime(7)
      dimension hlat(600),hlon(600),hdep(600),hmag(600)
      data isz_h/600/
      real*8 wprd
      dimension pstlat_seis(500), pstlon_seis(500)
!-----------------------------------------------------------------------
      real *8 epdist
!-----------------------------------------------------------------------
      call tttabl3(trp(1,1,1),trs(1,1,1),trp(1,1,2),trs(1,1,2))
      call getllst22(nllst,llst)
      call sttab3d(  &
          sttlat(1),sttlon(1),stthei(1),as(1),bs(1),cs(1), &
          ntat(1),jntat(1),kankn(1))
!------------------------------------------------------------------
      call lbppss('mk_dataset_seismic_tsunami.ps','A4P')
!------------------------------------------------------------------
!      read station data
!------------------------------------------------------------------
      call read_sta(nst_t,istno_t,flat_t,flon_t,isz_st_t)
!------------------------------------------------------------------
!      read tsunami data
!------------------------------------------------------------------
      call read_tsunami(isz_th,isz_tst,nth,ith_time(1,1),th_lat(1),th_lon(1), &
               th_dep(1),th_mag(1),th_mw(1),ntst(1), ist_no(1,1), &
               tst_lat(1,1),tst_lon(1,1),tst_high(1,1))
!------------------------------------------------------------------
!      seismic data list
!------------------------------------------------------------------
!     call system('ls -1 ../toyama/datasets/mag/datasets_ud/*.UD > band_pass_amp.list')
      call system('find ../toyama/datasets/mag/datasets_ud -type f -print > band_pass_amp.list')
      open(10,file='band_pass_amp.list',status='old')
      iend=0
      nh=0
      nfile=0
      do while(iend.eq.0)
         read(10,'(a80)',end=10) fname(nfile+1)
         nfile=nfile+1
!        write(6,'('' nfile='',i8,'' fname='',a40)') nfile,fname(nfile)
         call rd_wfile(fname(nfile),ihtime(1,nh+1), &
             hlat(nh+1),hlon(nh+1),hdep(nh+1),hmag(nh+1), &
             iwvtime(1),iwv(1),nwv,isz_wv, &
             fstlat(nfile),fstlon(nfile),sthigh,israte,galpc)
!        write(6,'('' nh='',i4)') nh
         if(nh.ge.1) then
            i_same=0
            do ih=1,nh
!              write(6,'('' ih='',i4,'' ihtime='',7i4)') &
!                         ih,(ihtime(i,ih),i=1,7)
               call prdtm7(ihtime(1,ih),ihtime(1,nh+1),wprd)
               if(abs(wprd).lt.2.1) then
                  ihnum(nfile)=ih
                  i_same=1
                  goto 30
               endif
            enddo
            if(i_same.eq.0) then
               nh=nh+1
               ihnum(nfile)=nh
               write(6,'('' nh='',i8,'' ihtime='',7i4)') &
                             nh,(ihtime(i,nh),i=1,7)
               if(nh.gt.isz_h) then
                  write(6,'('' nh>isz_h stop'')')
                  stop
               endif
            endif
  30        continue
         else
            nh=nh+1
            ihnum(nfile)=nh
         endif
      enddo
   10 continue
      close(10)
      write(6,'('' end of read seismic data'')')
!-------------------------------------------------------------
!      station map
!-------------------------------------------------------------
      xfl=150.0
      yfl=150.0
      xff=20.0
      yff=80.0
!
      flat1=20.0
      flat2=47.0
      flon1=120.0
      flon2=150.0
      flat0=(flat1+flat2)/2.0
      flon0=(flon1+flon2)/2.0
      xoff=0.0
      yoff=0.0
      scl=1.0
      theta=0.0
      mthd=9
      write(6,'('' flat0='',f10.3,'' flon0='',f10.3)') flat0,flon0
      
      CALL PRJCTN(FLAT0,FLON0,flat0,flon1,xw,yw, &
       &          XOFF,YOFF,SCL,THETA,MTHD,1)
      dish=sqrt(xw**2+yw**2)/1.0e6*2.0
!     write(6,'('' dish='',f10.3)') dish
      scl=scl*sqrt(xw**2+yw**2)/(xfl/2.0)
!     write(6,'('' scl='',f20.1)') scl
!--------------------------------------------------
!    map parameters
!--------------------------------------------------
      IPFRAM=4
      IPLLL=1
      IPSCL=0
      IPSHOR=4
      IPPREF=2
      IPPREF=0
      IPCRSS=0
      IPCONT=0
      IP76=0
      IP67=0
      IPOTH=0
!
      XL=xfl
      YL=yfl
      FLATIN=((flat2-flat1)+(flon2-flon1))/3.0*60.0
!     write(6,'('' (1) flatin='',f10.4)') flatin
        if(flatin.ge.1200.0) then
           flatin=1200.0
        elseif(flatin.ge.600.0) then
           flatin=600.0
        elseif(flatin.ge.300.0) then
           flatin=300.0
        elseif(flatin.ge.120.0) then
           flatin=120.0
        elseif(flatin.ge.60.0) then
           flatin=60.0
        elseif(flatin.ge.30.0) then
           flatin=30.0
        elseif(flatin.ge.20.0) then
           flatin=20.0
        elseif(flatin.ge.10.0) then
           flatin=10.0
        elseif(flatin.ge.5.0) then
           flatin=5.0
        elseif(flatin.ge.2.0) then
           flatin=2.0
        else
           flatin=1.0
        endif
!     write(6,'('' (2) flatin='',f10.4)') flatin
      FLONIN=flatin
      CMNT=' '
      MAPKwk='J'
      MAPK=mapkwk
      MTHD=8
!--------------------------------------------------
      xl=xfl
      yl=yfl
      x0=xff+xl/2.0
      y0=yff+yl/2.0
!     write(6,'('' xl='',f8.1,'' yl='',f8.1,'' scl='',e10.3, &
!    &      '' x0='',f8.1,'' y0='',f8.1)')  &
!    &          xl,yl,scl,x0,y0
!---------------------------------------------------------------
!     plot map
!---------------------------------------------------------------
      ipalt=0
      if(ipalt.ne.0) call pltalt
      call newpen(1)
      call setcol(0.0,0.0,0.0)
      IF(IPFRAM.NE.0) CALL PFRAME
      IF(IPLLL.NE.0) CALL PLLLIN
      mapk=mapkwk
      IF(IPSHOR.NE.0) CALL PSHRLN
      mapk='A'
      CALL PSHRLN
      chsize=3.0
!--------------------------------------------------------------
      ssize=1.2
      call newpen(1)
      call setcol(1.0,0.0,0.0)
      do ist=1,nst_t
         CALL PRJCTN(FLAT0,FLON0,flat_t(ist),flon_t(ist),xw,yw, &
                  XOFF,YOFF,SCL,THETA,MTHD,1)
         call fcircl(x0+xw,y0+yw,ssize/2.0)
      enddo
      call plot(0.0,0.0,999)
!------------------------------------------------------------------
!    tsunami height
!------------------------------------------------------------------
      nph=0
      do ith=1,nth
         if(ntst(ith).le.0) goto 70
         if(th_lat(ith).lt.20.0.or.th_lat(ith).gt.45.0) goto 70
         if(th_lon(ith).lt.120.0.or.th_lon(ith).gt.150.0) goto 70
         nstp=0
         hmax=0.0
         do ist=1,ntst(ith)
            do ist_t=1,nst_t
               if(ist_no(ith,ist).eq.istno_t(ist_t)) then
                  wstlat=flat_t(ist_t)
                  wstlon=flon_t(ist_t)
                  goto 20
               endif
            enddo   !  do ist_t=1,nst_t
            goto 50
   20       continue
            dlt=epdist(dble(th_lat(ith)),dble(th_lon(ith)), &
                       dble(flat_t(ist_t)),dble(flon_t(ist_t)))
            if(wstlat.ge.20.0.and.wstlat.le.45.0.and. & 
               wstlon.ge.120.0.and.wstlon.le.150.0.and.dlt.lt.500.0) then
               nstp=nstp+1
               pstlat(nstp)=wstlat
               pstlon(nstp)=wstlon
               psthigh(nstp)=tst_high(ith,ist)
               if(hmax.lt.psthigh(nstp)) hmax=psthigh(nstp)
            endif
   50       continue
         enddo   !  do ist=1,ntst(ith)
         if(nstp.ge.5.and.hmax.ge.50.0.and.th_mw(ith).ge.6.5.and.&
                th_lat(ith).ge.28.0) then
!           call feed
            write(fname_ps,'(i4.4,i2.2,i2.2,''_'',i2.2,i2.2,i2.2, &
                  & ''_M'',f3.1,''.ps'')') &
                       (ith_time(i,ith),i=1,6),th_mag(ith)
            call lbppss(fname_ps,'A4P')
            write(fname_out,'(i4.4,i2.2,i2.2,''_'',i2.2,i2.2,i2.2, &
                  & ''_M'',f3.1,''.4band.out'')') (ith_time(i,ith),i=1,6),th_mag(ith)
            open(15,file=fname_out,status='unknown')
            write(fname_out2,'(i4.4,i2.2,i2.2,''_'',i2.2,i2.2,i2.2, &
                  & ''_M'',f3.1,''.3band.out'')') (ith_time(i,ith),i=1,6),th_mag(ith)
            open(16,file=fname_out2,status='unknown')
!---------------------------------------------------------------------
!----------------------------------------------------------------------------
!          seismic wave
!----------------------------------------------------------------------------
            do ih=1,nh
               write(6,'('' ih='',i4)') ih
               call prdtm7(ith_time(1,ith),ihtime(1,ih),wprd) 
               if(abs(wprd).ge.60.0) goto 40
               dlt=epdist(dble(th_lat(ith)),dble(th_lon(ith)), &
                          dble(hlat(ih)),dble(hlon(ih)))
               if(dlt.gt.20.0) goto 40
!-------------------------------------------------------------------------
               write(15,'(i4.4,i2.2,i2.2,''_'',i2.2,i2.2,i2.2, &
                 & '' Lat:'',f7.4,'' Lon:'',f8.4,'' Dep:'',f5.1, &
                 & '' Mag:'',f5.2)') &
                 (ihtime(i,ih),i=1,6),hlat(ih),hlon(ih),hdep(ih),hmag(ih)
               write(16,'(i4.4,i2.2,i2.2,''_'',i2.2,i2.2,i2.2, &
                 & '' Lat:'',f7.4,'' Lon:'',f8.4,'' Dep:'',f5.1, &
                 & '' Mag:'',f5.2)') &
                 (ihtime(i,ih),i=1,6),hlat(ih),hlon(ih),hdep(ih),hmag(ih)
               ndx=0
               do ifile=1,nfile
                  if(ihnum(ifile).eq.0) then
                     write(6,'('' ihnum==0 ih='',i8,'' ifile='',i8,'' stop'')')&
                                 ih,ifile
                     stop
                  elseif(ihnum(ifile).eq.ih) then
                     dltmin=1.0e10
                     istp_near=0
                     do istp=1, nstp
                        dlt=epdist(dble(pstlat(istp)),dble(pstlon(istp)), &
                                   dble(fstlat(ifile)),dble(fstlon(ifile)))
                        if(dlt.lt.dltmin) then
                           dltmin=dlt
                           h_tsunami=psthigh(istp)
                        endif
                     enddo
!
                     if(dltmin.lt.30.0) then
                        ndx=ndx+1
                        dlt_sort(ndx)=epdist(dble(hlat(ih)),dble(hlon(ih)), &
                                dble(fstlat(ifile)),dble(fstlon(ifile)))
                        idx_sort(ndx)=ndx
                        idx_file(ndx)=ifile
                        hval_tsunami(ndx)=h_tsunami
                     endif
                  endif
               enddo
               call qsortf4(ndx,dlt_sort(1), idx_sort(1),1)
!
!              do ifile=1,nfile
               np_sst=0
               do jdx=1,ndx
                  idx=idx_sort(jdx)
                  ifile=idx_file(idx)
!
                  write(6,'('' ih='',i8,'' ifile='',i8)') ih,ifile
                  call rd_wfile(fname(ifile),iwhtime(1), &
                           whlat,whlon,whdep,whmag, &
                           iwvtime(1),iwv(1),nwv,isz_wv,stlat,stlon,sthigh, &
                           israte,galpc)
                  call prdtm7(ihtime(1,ih),iwhtime(1),wprd)
                  if(abs(wprd).gt.0.1) then
                     write(6,'('' iwhtime='',7i4,'' ihtime='',7i4,'' stop'')') &
                        iwhtime,(ihtime(i,ih),i=1,7)
                     stop
                  endif
                  do idat=1,nwv
                     wvdata(idat)=dble(iwv(idat))*galpc
                  enddo
                  call cal_band_amp(fname(ifile), &
                             nwv,israte,wvdata(1),isz_wv, &
                             iwvtime(1),stlat,stlon,ihtime(1,ih), &
                             hlat(ih),hlon(ih),hdep(ih),hmag(ih),galpc, &
                             hval_tsunami(idx),if_fail)
                  if(if_fail.eq.0) then
                     np_sst= np_sst+1
                     pstlat_seis(np_sst)=stlat
                     pstlon_seis(np_sst)=stlon
                  endif
               enddo
!-----------------------------------------------------------------------------------
               call feed
               call newpen(1)
               call setcol(0.0,0.0,0.0)
               IF(IPFRAM.NE.0) CALL PFRAME
               IF(IPLLL.NE.0) CALL PLLLIN
               mapk=mapkwk
               IF(IPSHOR.NE.0) CALL PSHRLN
               mapk='A'
               CALL PSHRLN
               chsize=2.5
!----------- tsunami station
               call newpen(1)
               call setcol(1.0,0.0,0.0)
               hh=30.0
               ww=1.0
               ssize=2.5
               do istp=1, nstp
                  CALL PRJCTN(FLAT0,FLON0,pstlat(istp),pstlon(istp),xw,yw, &
                     XOFF,YOFF,SCL,THETA,MTHD,1)
                  hval=hh*psthigh(istp)/hmax
                  call prect(x0+xw-ww/2.0,y0+yw,x0+xw+ww/2.0,y0+yw+hval)
               enddo
               CALL PRJCTN(FLAT0,FLON0,th_lat(ith),th_lon(ith),xw,yw, &
                     XOFF,YOFF,SCL,THETA,MTHD,1)
               call ffcrcl(x0+xw,y0+yw,ssize/2.0)
               call prect(x0+xl/2.0+10.0-ww/2.0,y0-yl/2.0,x0+xl/2.0+10.0+ww/2.0,y0-yl/2.0+hh)
               call newpen(1)
               call setcol(0.0,0.0,0.0)
               write(buf100,'(i4,''cm'')') int(hmax)
               call symbol(x0+xl/2.0+10.0+ww,y0-yl/2.0,chsize,buf100,0.0,6)
!---------- seismic station
               call newpen(1)
               call setcol(0.0,0.0,1.0)
               ssize=1.2
               do ip_sst=1, np_sst
                  CALL PRJCTN(FLAT0,FLON0,pstlat_seis(ip_sst),pstlon_seis(ip_sst),xw,yw, &
                     XOFF,YOFF,SCL,THETA,MTHD,1)
                  call ffcrcl(x0+xw,y0+yw,ssize/2.0)
               enddo
!
               call newpen(1)
               call setcol(0.0,0.0,0.0)
               write(buf100,'(i4.4,''/'',i2.2,''/'',i2.2,'' '',i2.2,'':'',i2.2,'':'',i2.2, &
                    & '' Mw:'',f3.1,'' Mjma:'',f3.1)') &
                   (ith_time(i,ith),i=1,6),th_mw(ith),th_mag(ith)
               call symbol(x0-xl/2.0,y0+yl/2.0+chsize/2.0,chsize,buf100,0.0,50)
               call plot(0.0,0.0,999)
               close(15)
               close(16)
   40          continue
            enddo
         endif   !  if(nstp.ge.5) then
   70    continue
      enddo   !  do ith=1,nth
!------------------------------------------------------------------
!     call plot(0.0,0.0,999)
      
      stop
      end
!==================================================
!     
      subroutine read_sta(nst_t,istno_t,flat_t,flon_t,isz_st_t)
!
!==================================================
      integer:: istno_t(isz_st_t)
      real:: flat_t(isz_st_t),flon_t(isz_st_t)
!------------------------------------------------------
      character buf100*100

      open(12,file='stat_j.txt.euc',status='old')
      iend=0
      do i=1,4
         read(12,'(a)') buf100
      enddo
!Ä¬°Ì´ÑÂ¬ÅÀ                    °ÞÅÙ   ·ÐÅÙ   ½êÂ°µ¡´Ø                   Í½Êó¶è¥³¡¼¥É
!¥³¡¼¥É Ì¾¾Î                   ÅÙ Ê¬  ÅÙ Ê¬                             1952 1957 1958 1968 1999
!10001  ¶üÏ©                   42 59 144 22  µ¤¾ÝÄ£                      062  042  022  002  100
!----+----|----+----|----+----|----+----|----+----|----+----|----+----|----+----|
      nst_t=0
      do while(iend.eq.0)
         read(12,'(a100)',end=10) buf100
         read(buf100,'(i5,24x,i3,i3,i4,i3)',err=20) ista,ilat,ilatm,ilon,ilonm
         nst_t=nst_t+1
         istno_t(nst_t)=ista
         flat_t(nst_t)=real(ilat)+real(ilatm)/60.0
         flon_t(nst_t)=real(ilon)+real(ilonm)/60.0
         if(nst_t.ge.isz_st_t) goto 10
   20    continue
      enddo
   10 continue
      close(12)
      write(6,'('' nst_t='',i4)') nst_t
      return
      end
!====================================================================
!
      subroutine read_tsunami(isz_th,isz_tst,nth,ith_time,th_lat,th_lon,th_dep,th_mag,th_mw,&
               ntst, ist_no,tst_lat,tst_lon,tst_high)
!
!====================================================================
      integer:: ith_time(7,isz_th)
      real:: th_lat(isz_th),th_lon(isz_th),th_dep(isz_th),th_mag(isz_th),th_mw(isz_th)
      integer:: ntst(isz_th),ist_no(isz_th,isz_tst)
      real:: tst_lat(isz_th,isz_tst),tst_lon(isz_th,isz_tst)
      real:: tst_high(isz_th,isz_tst)
!--------------------------------------------------------------------
      character buf100*100
!--------------------------------------------------------------------
      open(13,file='dat97_19.txt.euc',status='old')
      nth=0
!     ntst
      iend=0
      do while(iend.eq.0)
         read(13,'(a100)',end=10) buf100
!----+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8
!TE199704211 9   22004022053023210122040361003  16 0  7777          -122 1663 3077U
         if(buf100(1:2).eq.'TE') then
            if(nth.ge.isz_th) goto 10
            nth=nth+1
            read(buf100,'(53x,f2.1)') th_mw(nth)
!           write(6,'('' th_mw='',f5.2)') th_mw(nth)
            ith_time(1,nth)=9999
            ntst(nth)=0
!----+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8
!A199704212102264     -123504     1664056     33     79S64B     39
         elseif(buf100(1:1).eq.'A') then
!           write(6,'('' buf100='',a100)') buf100
            read(buf100,'(bz,1x,i4,i2,i2,i2,i2,i2,i2,4x,f3.0,f4.2,4x,f4.0,f4.2,4x,f5.2,3x,f2.1)') &
                (ith_time(i,nth),i=1,7), th_lat(nth),flatm,th_lon(nth),flonm, &
                th_dep(nth),th_mag(nth)
            ith_time(7,nth)=ith_time(7,nth)*10
            wlat= abs(th_lat(nth))+flatm/60.0
            if(th_lat(nth).gt.0.0) then
               th_lat(nth)=wlat
            else
               th_lat(nth)=-wlat
            endif
            wlon= abs(th_lon(nth))+flonm/60.0
            th_lon(nth)= abs(th_lon(nth))+flonm/60.0
            if(th_lon(nth).gt.0.0) then
               th_lon(nth)=wlon
            else
               th_lon(nth)=-wlon
            endif
!           write(6,'(7i4,f9.4,f9.4,f8.2,f6.2,f6.2)') &
!             (ith_time(i,nth),i=1,7), &
!             th_lat(nth),th_lon(nth),th_dep(nth),th_mag(nth),th_mw(nth)
            ntst(nth)=0
         elseif(buf100(1:1).eq.'I') then
!----+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8
!I100010021                                   221008       7 33 221008      11 33
!I100020021         220500220549U08470553   3 2210200431   8 29 2210200431  14 29
!           write(6,'('' buf100='',a100)') buf100
            if(buf100(63:63).eq.' '.and.ntst(nth).lt.isz_tst) then
               ntst(nth)= ntst(nth)+1
!              write(6,'('' nth='',i4,'' ntst='',i4)') nth,ntst(nth)
               read(buf100,'(1x,i5,49x,i4)') istn,iheight
               ist_no(nth,ntst(nth))=istn
               tst_high(nth,ntst(nth))=iheight
!              write(6,'('' ntst='',i4,'' ist_no='',i6,'' th='',f8.2)') &
!                 ntst(nth),ist_no(nth,ntst(nth)),tst_high(nth,ntst(nth))
            endif
         endif
      enddo 
   10 continue
      close(13)
      do ith=1,nth
         write(6,'('' ith='',i5,'' ot='',i4,6i3,'' ntst='',i5,'' Mw='',f5.2)') &
             ith,(ith_time(i,ith),i=1,7),ntst(ith),th_mw(ith)
      enddo
      return
      end
!===========================================================
!
      subroutine rd_wfile(fname,ihtime,hlat,hlon,hdep,hmag, &
            iwvtime,iwv,nwv,isz_wv,stlat,stlon,sthigh,israte,galpc)
!
!===========================================================
      character fname*80,chw100*100
      dimension iwv(isz_wv)
      dimension ihtime(7),iwvtime(7),ictime(7)
!-----------------------------------------------------------
!     write(6,'('' fname='',a40)') fname
      open(11,file=fname,status='old')
      jend=0
      do while(jend.eq.0) 
         read(11,'(a40)') chw100
!        write(6,'('' chw100='',a50)') chw100
! ----+----|----+----|----+----|----+----|
! Origin Time       2012/12/07 17:18:00
! Lat.              38.018
! Long.             143.867
! Depth. (km)       49
! Mag.              7.3
! Station Code      IWT016
! Station Lat.      39.5997
! Station Long.     141.6789
! Station Height(m) 198
! Record Time       2012/12/07 17:19:05
! Sampling Freq(Hz) 100Hz
! Duration Time(s)  185
! Dir.              U-D
! Scale Factor      3920(gal)/6182761
! Max. Acc. (gal)   31.582
! Last Correction   2012/12/07 17:19:05
! Memo.             
! ----+----|----+----|----+----|----+----|----+----|----+----|----+----|
!     6699     6650     6612     6597     6572     6535     6525     6542
         if(chw100(1:18).eq.'Origin Time       ') then
         read(chw100(19:50),'(i4,1x,i2,1x,i2,1x,i2,1x,i2,1x,i2)') &
             (ihtime(i),i=1,6)
         ihtime(7)=0
!        write(6,'('' ihtime='',7i4)') ihtime
         endif
         if(chw100(1:18).eq.'Lat.              ') then
         read(chw100(19:50),'(f10.0)') hlat
         endif
         if(chw100(1:18).eq.'Long.             ') &
           read(chw100(19:50),'(f10.0)') hlon
         if(chw100(1:18).eq.'Depth. (km)       ') &
           read(chw100(19:50),'(f10.0)') hdep
         if(chw100(1:18).eq.'Mag.              ') then
            read(chw100(19:50),'(f10.0)') hmag
!           write(6,'('' hlat='',f8.3,'' hlon='',f8.3,'' hdep='',f6.1, &
!              & '' hmag='',f5.2)') hlat,hlon,hdep,hmag
         endif
         if(chw100(1:18).eq.'Station Code      ') &
           read(chw100(19:50),'(a6)') stcode
         if(chw100(1:18).eq.'Station Lat.      ') &
           read(chw100(19:50),'(f10.0)') stlat
         if(chw100(1:18).eq.'Station Long.     ') then
         read(chw100(19:50),'(f10.0)') stlon
         endif
         if(chw100(1:18).eq.'Station Height(m) ') then
            read(chw100(19:50),'(f10.0)') sthigh
!           write(6,'('' stlat='',f8.3,'' stlon='',f8.3,'' sthigh='', &
!             & f8.2)') stlat,stlon,sthigh
         endif
         if(chw100(1:18).eq.'Record Time       ') then
         read(chw100(19:50),'(i4,1x,i2,1x,i2,1x,i2,1x,i2,1x,i2)') &
              (iwvtime(i),i=1,6)
         iwvtime(7)=0
!        write(6,'('' iwvtime='',7i4)') iwvtime
         endif
         if(chw100(1:18).eq.'Sampling Freq(Hz) ') &
           read(chw100(19:50),'(i3)') israte
         if(chw100(1:18).eq.'Duration Time(s)  ') &
           read(chw100(19:50),'(i3)') itlen
         if(chw100(1:18).eq.'Dir.              ') &
           read(chw100(19:50),'(a3)') chcomp
         if(chw100(1:18).eq.'Scale Factor      ') then
         do i=19,50
            if(chw100(i:i+5).eq.'(gal)/') then
               read(chw100(19:i-1),*) igal
               read(chw100(i+6:i+20),*) icount
               galpc=real(igal)/real(icount)
               goto 20
            endif
         enddo
   20          continue
!              read(chw100(19:50),'(i4,6x,i7)') igal,icount
!        write(6,'('' igal='',i10,'' icount='',i10)') igal,icount
         endif
         if(chw100(1:18).eq.'Max. Acc. (gal)   ') &
           read(chw100(19:50),'(f10.0)') accmax
         if(chw100(1:18).eq.'Last Correction   ') then
         read(chw100(19:50),'(i4,1x,i2,1x,i2,1x,i2,1x,i2,1x,i2)') &
              (ictime(i),i=1,6)
         ictime(7)=0
!        write(6,'('' ictime='',7i4)') ictime
         endif
         if(chw100(1:18).eq.'Memo.             ') then
         do i=1,isz_wv
            iwv(i)=0
         enddo
         nwv=itlen*israte
!        write(6,'('' itlen='',i8,'' israte='',i8,'' nwv='',i10)') &
!            itlen,israte,nwv
         if(nwv.gt.isz_wv) nwv=isz_wv
         nline=(nwv-1)/8+1
         do iline=1,nline
            i1=(iline-1)*8+1
            i2=(iline-1)*8+8
            if(i2.gt.nwv) i2=nwv
!                 write(6,'('' i1='',i8,'' i2='',i8)') i1,i2
            read(11,'(i8,7(1x,i8))') (iwv(i),i=i1,i2)
         enddo
         goto 30
         endif
      enddo
   30 continue
      close(11)
      return
      end
!================================================================
!
      subroutine cal_band_amp(fname,ndat,israte,wvdata,iszwvd_in, &
                    idstime,stlat,stlon,ihtime, &
                    hlat,hlon,hdep,hmag,galpc, &
                    hval_tsunami,if_fail)
!
!================================================================
      character fname*80
      real*8 wvdata(iszwvd_in)
      dimension idstime(7),ihtime(7)
!-------------------------------------------------------
      real trp,trs ! pp/ss 
      common /pp/trp(DISDIM,DEPDIM,2)
      common /ss/trs(DISDIM,DEPDIM,2)
      integer nllst,llst ! llst
      common /llst/nllst,llst(1000)
!------ station info.---------------------------------------------
      character ntat*6,jntat*8 ! m2
      common /m2/ntat(MAXS),jntat(MAXS)
      real*8  as,bs,cs,sttlat,sttlon,stthei ! m1
      integer kankn                  ! m1
      common /m1/as(MAXS),bs(MAXS),cs(MAXS),  &
                 sttlat(MAXS),sttlon(MAXS),stthei(MAXS),  &
                 kankn(MAXS)
!-------------------------------------------------------
      data iszwvd/600000/
      real*8 wvdata2(600000)
      real*8 wsum
      real*8 dgn,dhfilt(100),dgn_tmp, dflk,dfhk,dap,ddt
      dimension vdmax(3,10)
      dimension xpyln(600000),ypyln(600000)
      dimension wave(3)
      character chw100*100
      dimension tmpos(100)
      real*8 stime
      real*8 dhdep
      data pi/3.1415926/
!-------------------------------------------------------------
      real *8 epdist
!-------------------------------------------------------------
      srate=israte
      write(6,'('' cal_band_amp srate='',f8.3,'' ndat='',i8)') srate,ndat
      if(ndat.lt.int(srate*60))  then
         write(6,'('' ndat < 60 sec'')')
         return
      endif
!     srate=israte
      nwdata=ndat
      if(nwdata.gt.iszwvd) nwdata=iszwvd
      ddt=1.0d0/srate
      write(6,'('' ndat='',i6,'' dt='',f6.3)') nwdata,ddt
!     call feed
      nx=1
      ny=7
      xpoff=20.0
      ypoff=15.0
      xlen=200.0-xpoff
      ylen=270.0-ypoff*2.0
      xfl=xlen/real(nx)*0.9
      yfl=ylen/real(ny)*0.65
      ipos=0
!---------------------------------------------------------
      tleng=real(nwdata)/srate
      timscl=xfl/tleng
!---------------------------------------------------------
!      filt
!---------------------------------------------------------
      wmax=0.0
      do icomp=3,3
!------- remove off set 
         wsum=0.0d0
         nsum=nwdata
         do idat=1,nsum
            wsum=wsum+wvdata(idat)
         enddo
         wave(icomp)=wsum/dble(nsum)
         write(6,'('' cal_si5 icomp='',i4,'' wave='',f8.3)') &
                   icomp,wave(icomp)
         do idat=1,nwdata
            wvdata2(idat)=wvdata(idat)-wave(icomp)
            if(wmax.lt.abs(wvdata2(idat))) &
                  wmax=abs(wvdata2(idat))
         enddo
      enddo
      chsize=2.0
      call c7t1(idstime(1),stime)
      do icomp=3,3
         ipos=ipos+1
         xff=xpoff + real(mod(ipos-1,nx))/real(nx)*xlen
         yff=ypoff + real(ny-(ipos-1)/nx-1)/real(ny)*ylen
!
         itick=0
         call tsdtsc(xfl,xff,yff,stime,timscl,chsize,itick, &
    &                         tmpos(1),ntmpos)
         call prect(xff,yff,xff+xfl,yff+yfl)
         if(icomp.eq.1) then
            write(chw100,'(''Lat:'',f7.3,'' Lon:'',f8.3)') stlat,stlon
            call symbol(xff,yff+yfl+chsize/2.0,chsize,chw100,0.0,30)
         endif
         write(chw100,'(e9.2)') wmax
         call symbol(xff-chsize*10.0,yff+yfl-chsize,chsize,chw100,0.0,9)
         write(chw100,'(e9.2)') -wmax
         call symbol(xff-chsize*10.0,yff,chsize,chw100,0.0,9)
         npyln=0
         do idat=1,nwdata
            xx=xff+real(idat-1)/srate*timscl
            yy=yff+yfl/2.0+wvdata2(idat)/wmax*yfl/2.0
            npyln=npyln+1
            xpyln(npyln)=xx
            ypyln(npyln)=yy
         enddo
         call plines(xpyln(1),ypyln(1),npyln)
      enddo
!--------- hypo info. ----------------------------------------------
      call symbol(xff,yff+yfl+chsize*3.5,chsize,fname,0.0,50)
      write(chw100,'(''H:'',i4.4,''/'',i2.2,''/'',i2.2,'' '', &
             & i2.2,'':'',i2.2,'':'',i2.2,'' Lat:'',f7.3, &
             & '' Lon:'',f8.3,'' Dep:'',f5.1,'' M:'',f5.2)') &
            (ihtime(i),i=1,6),hlat,hlon,hdep,hmag
      chsize=2.0
      call symbol(xff,yff+yfl+chsize*2.0,chsize,chw100,0.0,80)
      dlt=epdist(dble(hlat),dble(hlon),dble(stlat),dble(stlon))
      write(chw100,'('' Stlat:'',f7.3,'' Stlon:'',f8.3, &
             & '' Dlt:'',f5.1,''km'')') &
          stlat,stlon,dlt
      call symbol(xff,yff+yfl+chsize/2.0,chsize,chw100,0.0,80)
!--------- travel time --------------------------------------------
      dhdep = hdep
      ist=60
      method=1
      ips=1
      call trtpsh4(dlt,dhdep,ist,method,ips,trp(1,1,1),trs(1,1,1), &
               trp(1,1,2),trs(1,1,2),kankn,nllst,llst(1),trcp,dtr,dth)
      ips=2
      call trtpsh4(dlt,dhdep,ist,method,ips,trp(1,1,1),trs(1,1,1), &
               trp(1,1,2),trs(1,1,2),kankn,nllst,llst(1),trcs,dtr,dth)
      call prdtm7(idstime(1),ihtime(1),wprd)
      p_time=wprd+trcp
      s_time=wprd+trcs
      f_time=wprd+trcs*2.0+60.0
!------------------------------------------------------------------
      if_fail=0
      do ifilt=1,4
!
         dhfilt(1)=1.0
         dhfilt(2)=0.0
         dhfilt(3)=-1.0
         dhfilt(4)=0.0
         dhfilt(4+1)=1.0
         dhfilt(4+2)=0.0
         dhfilt(4+3)=-1.0
         dhfilt(4+4)=0.0
         m_filt=2
         dgn=ddt/2.0d0 * ddt/2.0d0
!------ band pass filter
         if(ifilt.eq.1) then
            ts=1.0
            tl=3.0
            dfhk=1.0d0/ts
            dflk=1.0d0/tl
         elseif(ifilt.eq.2) then
            ts=3.0
            tl=9.0
            dfhk=1.0d0/ts
            dflk=1.0d0/tl
         elseif(ifilt.eq.3) then
            ts=9.0
            tl=27.0
            dfhk=1.0d0/ts
            dflk=1.0d0/tl
         else
            ts=27.0
            tl=81.0
            dfhk=1.0d0/ts
            dflk=1.0d0/tl
         endif
         nbes=4
         dap=1.0d0
         call BESPSN_d(dhfilt(m_filt*4+1),m,dgn_tmp,nbes, &
                               dflk,dfhk,dap,ddt,IERR)
!        write(6,'('' BESPSN_d ierr='',i8)') ierr
         m_filt= m_filt+m
!        write(6,'('' m='',i4,'' dgn_tmp='',e10.3,'' dhfilt='',30f10.3)') &
!            m,dgn_tmp,(dhfilt(i),i=1,m_filt*4)
!        write(6,'('' m='',i4,'' m_filt='',i4)') m,m_filt
!        write(6,'('' nwdata='',i8)') nwdata
         dgn= dgn*dgn_tmp
         do icomp=3,3
            vdmax(icomp,ifilt)=0.0
            do idat=1,nwdata
               wvdata2(idat)=wvdata(idat)-wave(icomp)
            enddo
!           if(ifilt.eq.2) then
!              write(6,'(''B idat='',i8,'' wave='',e9.2,'' wvdata='',e9.2,&
!               '' wvdata2='',e9.2)') &
!                 (idat,wave(icomp),wvdata(idat),wvdata2(idat), &
!                  idat=1,nwdata)
!           endif
            call TANDEM_d(wvdata2(1),wvdata2(1),nwdata, &
                      dhfilt(1),m_filt,dgn,1)
!           if(ifilt.eq.2) then
!              write(6,'(''A idat='',i8,'' wave='',e9.2,'' wvdata='',e9.2,&
!               '' wvdata2='',e9.2)') &
!                 (idat,wave(icomp),wvdata(idat),wvdata2(idat), &
!                  idat=1,nwdata)
!           endif
            idat_max=1
            do idat=1,nwdata
               if(abs(wvdata2(idat)).gt.  vdmax(icomp,ifilt)) then
                  vdmax(icomp,ifilt)=abs(wvdata2(idat))
                  idat_max=idat
               endif
!              write(6,'('' icomp='',i4,'' ifilt='',i4,'' idat='',i8, &
!                 & '' vdmax='',e9.2)') &
!                   icomp,ifilt,idat,vdmax(icomp,ifilt)
            enddo
            t_max=real(idat_max-1)/srate
!           write(6,'('' icomp='',i4,'' ifilt='',i4,'' vdmax='',e9.2)') &
!                   icomp,ifilt,vdmax(icomp,ifilt)
            if(icomp.eq.3) then
               wmax=vdmax(icomp,ifilt)
               ipos=ipos+1
               xff=xpoff + real(mod(ipos-1,nx))/real(nx)*xlen
               yff=ypoff + real(ny-(ipos-1)/nx-1)/real(ny)*ylen
               itick=0
               call tsdtsc(xfl,xff,yff,stime,timscl,chsize,itick, &
    &                         tmpos(1),ntmpos)
               call prect(xff,yff,xff+xfl,yff+yfl)
               amplevel=galpc/(2.0*pi/tl)/(2.0*pi/tl)
               if(amplevel.gt.wmax) then
                  wmax=amplevel
                  call newpen(1)
                  call setcol(1.0,0.0,0.0)
                  xpyln(1)=xff
                  ypyln(1)=yff+yfl
                  xpyln(2)=xff+xfl
                  ypyln(2)=yff+yfl
                  call bpyln(xpyln(1),ypyln(1),2,1.0)
                  call newpen(1)
                  call setcol(0.0,0.0,0.0)
                  if_fail=1
               endif
               write(chw100,'(e9.2)') wmax
               call symbol(xff-chsize*10.0,yff+yfl-chsize,chsize, &
                             chw100,0.0,9)
               write(chw100,'(e9.2)') -wmax
               call symbol(xff-chsize*10.0,yff,chsize,chw100,0.0,9)
               npyln=0
               do idat=1,nwdata
                  xx=xff+real(idat-1)/srate*timscl
                  yy=yff+yfl/2.0+wvdata2(idat)/wmax*yfl/2.0
                  npyln=npyln+1
                  xpyln(npyln)=xx
                  ypyln(npyln)=yy
               enddo
               call plines(xpyln(1),ypyln(1),npyln)
!-----------p,s time
               xxp=xff+p_time*timscl
               xxs=xff+s_time*timscl
               xpyln(1)=xxp
               ypyln(1)=yff
               xpyln(2)=xxp
               ypyln(2)=yff+yfl
               call newpen(1)
               call setcol(1.0,0.0,0.0)
               call bpyln(xpyln(1),ypyln(1),2,1.0)
               call symbol(xxp+chsize/2.0,yff+chsize/2.0,chsize,'P',0.0,1)
               xpyln(1)=xxs
               xpyln(2)=xxs
               call newpen(1)
               call setcol(0.0,0.0,1.0)
               call bpyln(xpyln(1),ypyln(1),2,1.0)
               call symbol(xxs+chsize/2.0,yff+chsize/2.0,chsize,'S',0.0,1)
!----------peak amp check --------------------------------
               write(6,'(''if_fail='',i2)') if_fail
               if(t_max.lt.p_time) then
                  write(6,'('' t_max < p_time'')')
                  if_fail=2
               elseif(t_max.gt.f_time) then
                  write(6,'('' t_max > f_time'')')
                  if_fail=3
               elseif(tleng.lt.f_time) then
                  write(6,'('' tleng < f_time'')')
                  if_fail=4
               endif
               write(6,'(''t_max='',f6.1,'' p_time='',f6.1,'' s_time='', &
                   & f6.1,'' f_time='',f6.1,'' tleng='',f6.1, &
                   & '' if_fail='',i1)') &
                    t_max,p_time,s_time,f_time,tleng,if_fail
!------------ amp level ----------------------------------------------
               ylevel=amplevel/wmax*yfl/2.0
               write(6,'('' galpc='',e10.3,'' amplevel='',e10.3, &
                 & '' ylevel='',e10.3)') galpc,amplevel,ylevel
!              if(ylevel.lt.yfl/2.0) then
!                 call newpen(1)
!                 call setcol(1.0,0.0,0.0)
!                 xpyln(1)=xff
!                 ypyln(1)=yff+yfl/2.0+ylevel
!                 xpyln(2)=xff+xfl
!                 ypyln(2)=yff+yfl/2.0+ylevel
!                 call bpyln(xpyln(1),ypyln(1),2,1.0)
!                 ypyln(1)=yff+yfl/2.0-ylevel
!                 ypyln(2)=yff+yfl/2.0-ylevel
!                 call bpyln(xpyln(1),ypyln(1),2,1.0)
!              endif
               call newpen(1)
               call setcol(0.0,0.0,0.0)
               write(chw100,'(''Dmax:'',f8.4,''(cm) Pass:'',f4.1, &
                  & ''-'',f4.1,''s QNlvl:'',f8.4,''cm'')') &
                  wmax,ts,tl,amplevel
               call symbol(xff,yff+yfl+chsize/2.0,chsize,chw100,0.0,60)
            endif
         enddo  !  do icomp=3,3
      enddo
      icomp=3
      write(6,'(''CCC '',f8.4,'','',f8.4, &
           & '','',4(f10.5,'',''),i4,'','',i4,'','',i4,'','',i4)') &
           stlat,stlon,(vdmax(icomp,ifilt),ifilt=1,4), &
           ihtime(1), ihtime(2), ihtime(3),ihtime(4)
      if(if_fail.eq.0) then
         write(15,'(f6.3,'','',f7.3,'','',f8.4,'','',f8.4, &
           & '','',4(f10.5,'',''),i4,'','',i4.4,i2.2,i2.2,i2.2,i2.2)') &
           hlat,hlon,stlat,stlon,(vdmax(icomp,ifilt),ifilt=1,4),int(hval_tsunami+0.1), &
           (ihtime(i),i=1,5)
         write(16,'(f6.3,'','',f7.3,'','',f8.4,'','',f8.4, &
           & '','',3(f10.5,'',''),i4,'','',i4.4,i2.2,i2.2,i2.2,i2.2)') &
           hlat,hlon,stlat,stlon,(vdmax(icomp,ifilt),ifilt=1,3),int(hval_tsunami+0.1), &
           (ihtime(i),i=1,5)
      else
         call newpen(1)
         call setcol(0.0,1.0,0.0)
         xx=xff+f_time*timscl
         call plot(xx,yff,3)
         call plot(xx,yff+yfl,2)
         call newpen(1)
         call setcol(1.0,0.0,0.0)
         ipos=ipos+1
         xff=xpoff + real(mod(ipos-1,nx))/real(nx)*xlen
         yff=ypoff + real(ny-(ipos-1)/nx-1)/real(ny)*ylen
         if(if_fail.eq.1) then
            call symbol(xff,yff,chsize,'A_max < QN level',0.0,16)
         elseif(if_fail.eq.2) then
            call symbol(xff,yff,chsize,'T_max < P time',0.0,14)
         elseif(if_fail.eq.3) then
            call symbol(xff,yff,chsize,'T_max > F time',0.0,14)
         elseif(if_fail.eq.4) then
            call symbol(xff,yff,chsize,'T_len < F time',0.0,14)
         endif
         write(chw100,'(''t_max='',f6.1,'' p_time='',f6.1,'' s_time='', &
                   & f6.1,'' f_time='',f6.1,'' tleng='',f6.1, &
                   & '' if_fail='',i1)') &
                    t_max,p_time,s_time,f_time,tleng,if_fail
         call symbol(xff,yff+chsize*1.5,chsize,chw100,0.0,90)
         call newpen(1)
         call setcol(0.0,0.0,0.0)
      endif
!----------------------------------------------------------------
      call feed
!----------------------------------------------------------------
      return
      end
